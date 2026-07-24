"""
Stratified tile sampling and batch label generation for the coupled-STiC
NN emulator.

Pipeline:
  select_tiles()  candidate tiles on a stride grid (stride = tile-2*apron)
                  with random jitter (breaks phase-locking between the
                  tile grid and the coarse instrument grids), k-means
                  clustered on tile content, stratified draw.
  prepare_runs()  extract a self-contained run dir per selected tile.
  run_local()     run them sequentially with the local binary; or
  write_slurm()   emit a SLURM array script for the cluster.
  harvest()       read inverted tiles, strip the apron, compute per-region
                  chi2 QC, write one .npz record per tile + an index.

The NN dataloader later consumes the per-tile records directly (obs
windows at native resolution + label stratifications + geometry).
"""

import os
import re
import sys
import json
import subprocess
import numpy as np

from . import tiles
from . import degradation as dg

from .sampling import minibatch_kmeans, _assign, stratified_draw


# --------------------------------------------------------------------------- #
# candidate positions and content features
# --------------------------------------------------------------------------- #

def candidate_positions(model_ny, model_nx, tile, apron, rng, jitter=True):
    """Stride-grid tile origins with random jitter, clamped in-bounds."""
    stride = tile - 2 * apron
    xs = np.arange(0, model_nx - tile + 1, stride)
    ys = np.arange(0, model_ny - tile + 1, stride)
    # make sure the far edge is covered
    if xs[-1] != model_nx - tile:
        xs = np.append(xs, model_nx - tile)
    if ys[-1] != model_ny - tile:
        ys = np.append(ys, model_ny - tile)
    pos = np.array([(y, x) for y in ys for x in xs])
    if jitter:
        j = rng.integers(-apron, apron + 1, size=pos.shape)
        pos = np.clip(pos + j, 0, [model_ny - tile, model_nx - tile])
    return pos  # (ncand, 2) as (my0, mx0)


def _fine_region(src_dir):
    """The region observed on the fine grid (lts[0] < 0), for features."""
    with open(os.path.join(src_dir, 'input.cfg')) as f:
        regions = tiles.parse_regions(f.read())
    for reg in regions:
        o = tiles.CoupledObs(os.path.join(src_dir, reg['obs_file']))
        if int(o.lts[0]) < 0 and o.nw > 2:
            return o
    raise RuntimeError('no fine-grid multi-wavelength region found')


def tile_features(src_dir, positions, tile):
    """Per-candidate-tile feature vector from the fine-grid region.

    Uses the mean spectrum to locate continuum-like (max <I>) and
    core-like (min <I>) wavelengths, then summarizes each tile window
    with intensity statistics that separate quiet / magnetic / flaring
    tiles reasonably well.
    """
    o = _fine_region(src_dir)
    I = o.dat[0, :, :, :, 0]                        # (ny, nx, nw)
    used = o.weights[:, 0] < 1e10
    mI = np.nanmean(I[..., used], axis=(0, 1))
    wsel = np.where(used)[0]
    w_cont = wsel[np.argmax(mI)]
    w_core = wsel[np.argmin(mI)]
    cont = I[:, :, w_cont]
    core = I[:, :, w_core]

    feats = []
    for (my0, mx0) in positions:
        c = cont[my0:my0 + tile, mx0:mx0 + tile]
        k = core[my0:my0 + tile, mx0:mx0 + tile]
        feats.append([np.nanmean(c), np.nanstd(c), np.nanmin(c),
                      np.nanmean(k), np.nanstd(k), np.nanmax(k),
                      np.nanmax(k) / max(np.nanmean(k), 1e-30)])
    X = np.asarray(feats)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def select_tiles(src_dirs, tile, apron, n_tiles, n_clusters=8,
                 seed=1234, jitter=True):
    """Stratified tile selection over one or more source run dirs.

    Returns a list of dicts: {src, my0, mx0, cluster}.
    """
    rng = np.random.default_rng(seed)
    all_pos, all_src, all_X = [], [], []
    for si, src in enumerate(src_dirs):
        o = _fine_region(src)
        pos = candidate_positions(o.ny, o.nx, tile, apron, rng, jitter)
        X = tile_features(src, pos, tile)
        all_pos.append(pos)
        all_src.append(np.full(len(pos), si))
        all_X.append(X)
    pos = np.vstack(all_pos)
    srcs = np.concatenate(all_src)
    X = np.vstack(all_X)

    k = min(n_clusters, len(pos))
    if len(pos) <= n_tiles:
        sel = np.arange(len(pos))
        labels_ = np.zeros(len(pos), int)
    else:
        centers = minibatch_kmeans(X, k, rng, iters=100)
        labels_ = _assign(X, centers)
        sel = stratified_draw(labels_, n_tiles, 0.5, 1, rng)

    out = [dict(src=src_dirs[int(srcs[i])], my0=int(pos[i, 0]),
                mx0=int(pos[i, 1]), cluster=int(labels_[i])) for i in sel]
    print(f'select_tiles: {len(out)} tiles from {len(pos)} candidates '
          f'({k} clusters)')
    return out


# --------------------------------------------------------------------------- #
# batch preparation / execution
# --------------------------------------------------------------------------- #

def prepare_runs(selection, out_root, tile, cfg_overrides=None,
                 aux_search=()):
    """Extract one run dir per selected tile under out_root/tile_NNNN."""
    os.makedirs(out_root, exist_ok=True)
    overrides = dict(cfg_overrides or {})
    overrides.setdefault('output_atmos', 'atmosout_tile.nc')
    overrides.setdefault('output_profiles', 'synthetic_tile.nc')
    overrides.setdefault('verbose', '1')
    index = []
    for i, s in enumerate(selection):
        rd = os.path.join(out_root, f'tile_{i:04d}')
        tiles.extract_tile_rundir(s['src'], rd, s['mx0'], s['my0'],
                                  tile, tile, cfg_overrides=overrides,
                                  aux_search=aux_search)
        index.append(dict(run_dir=rd, **s))
        print(f'prepare_runs: {rd}  (my0={s["my0"]}, mx0={s["mx0"]})')
    with open(os.path.join(out_root, 'index.json'), 'w') as f:
        json.dump(dict(tile=tile, tiles=index), f, indent=1)
    return index


def run_local(out_root, stic_binary, mpiexec='mpiexec', nproc=8,
              only=None):
    """Run every prepared tile sequentially with the local binary."""
    with open(os.path.join(out_root, 'index.json')) as f:
        index = json.load(f)['tiles']
    for i, rec in enumerate(index):
        if only is not None and i not in only:
            continue
        rd = rec['run_dir']
        if os.path.isfile(os.path.join(rd, 'atmosout_tile.nc')):
            print(f'run_local: {rd} already done, skipping')
            continue
        print(f'run_local: [{i + 1}/{len(index)}] {rd}')
        with open(os.path.join(rd, 'stic_tile.log'), 'w') as log:
            ret = subprocess.call([mpiexec, '-n', str(nproc), stic_binary],
                                  cwd=rd, stdout=log,
                                  stderr=subprocess.STDOUT)
        if ret != 0:
            print(f'run_local: WARNING {rd} exited with {ret}')


def write_slurm(out_root, stic_binary, ntasks=16, time='02:00:00',
                partition=None, mpiexec='mpiexec'):
    """SLURM array script: one array element per tile run dir."""
    with open(os.path.join(out_root, 'index.json')) as f:
        n = len(json.load(f)['tiles'])
    part = f'#SBATCH --partition={partition}\n' if partition else ''
    script = f"""#!/bin/bash
#SBATCH --job-name=stic_tiles
#SBATCH --array=0-{n - 1}
#SBATCH --ntasks={ntasks}
#SBATCH --time={time}
{part}
cd {os.path.abspath(out_root)}/tile_$(printf '%04d' $SLURM_ARRAY_TASK_ID)
{mpiexec} -n {ntasks} {stic_binary} > stic_tile.log 2>&1
"""
    path = os.path.join(out_root, 'run_tiles.slurm')
    with open(path, 'w') as f:
        f.write(script)
    print(f'write_slurm: {path} ({n} array elements) — submit with sbatch')
    return path


# --------------------------------------------------------------------------- #
# harvesting
# --------------------------------------------------------------------------- #

def _region_chi2_map(run_dir, reg_file):
    """Per-observed-pixel reduced chi2 of one region from STiC's
    degraded output (out_<reg>.nc). Returns (chi2_map, act_mask)."""
    from netCDF4 import Dataset
    tob = tiles.CoupledObs(os.path.join(run_dir, reg_file))
    with Dataset(os.path.join(run_dir, 'out_' + reg_file)) as f:
        syn = np.ma.filled(f.variables['profiles'][0], np.nan)
    act = tob.pweights[0] > 0
    use = tob.weights < 1e10                       # (nw, ns)
    sig = np.where(use, tob.weights, np.inf)
    r = (tob.dat[0] - syn) / sig[None, None]
    r = np.where(np.isfinite(r), r, 0.0)
    ndata = max(int(use.sum()), 1)
    return (r * r).sum(axis=(2, 3)) / ndata, act


def _region_chi2(run_dir, reg_file):
    """Mean reduced chi2 over active pixels (backward-compatible)."""
    cmap, act = _region_chi2_map(run_dir, reg_file)
    if not act.any():
        return np.nan, act
    return float(cmap[act].mean()), act


def harvest(out_root, apron, atmos_name='atmosout_tile.nc'):
    """Collect labels from all completed tiles into labels/<tile>.npz."""
    with open(os.path.join(out_root, 'index.json')) as f:
        index = json.load(f)['tiles']
    lab_dir = os.path.join(out_root, 'labels')
    os.makedirs(lab_dir, exist_ok=True)

    summary = []
    for i, rec in enumerate(index):
        rd = rec['run_dir']
        if not os.path.isfile(os.path.join(rd, atmos_name)):
            print(f'harvest: {rd} not inverted yet, skipping')
            continue
        model, yy, xx = tiles.read_tile_labels(rd, atmos_name, apron)
        with open(os.path.join(rd, 'input.cfg')) as f:
            regions = tiles.parse_regions(f.read())

        rec_out = dict(
            src=rec['src'], my0=rec['my0'], mx0=rec['mx0'],
            apron=apron, y_coords=yy, x_coords=xx,
            ltau=model.ltau[0, 0, 0],
            temp=model.temp[0], vlos=model.vlos[0], vturb=model.vturb[0],
            blong=model.Bln[0], bhor=model.Bho[0], azi=model.azi[0])

        chi2s = {}
        for reg in regions:
            tob = tiles.CoupledObs(os.path.join(rd, reg['obs_file']))
            tag = os.path.splitext(reg['obs_file'])[0]
            rec_out[f'{tag}_dat'] = tob.dat[0].astype('float32')
            rec_out[f'{tag}_pweights'] = tob.pweights[0].astype('float32')
            rec_out[f'{tag}_wav'] = tob.wav
            rec_out[f'{tag}_weights'] = tob.weights.astype('float32')
            rec_out[f'{tag}_lts'] = tob.lts
            rec_out[f'{tag}_ltargs'] = tob.ltargs
            rec_out[f'{tag}_ds'] = tob.ds.astype('float32')
            if os.path.isfile(os.path.join(rd, 'out_' + reg['obs_file'])):
                c2, _ = _region_chi2(rd, reg['obs_file'])
                chi2s[tag] = c2
                rec_out[f'{tag}_chi2'] = c2
        np.savez_compressed(os.path.join(lab_dir, f'tile_{i:04d}.npz'),
                            **rec_out)
        summary.append((i, chi2s))
        print(f'harvest: tile_{i:04d}  labels {model.ny}x{model.nx}  '
              f'chi2 {chi2s}')

    with open(os.path.join(lab_dir, 'harvest_summary.json'), 'w') as f:
        json.dump([{'tile': i, 'chi2': c} for i, c in summary], f, indent=1)
    print(f'harvest: {len(summary)} tiles -> {lab_dir}')
    return lab_dir


# --------------------------------------------------------------------------- #
# full-map harvesting (use an existing, possibly partially converged,
# full-FOV coupled inversion as a label source — no tiles needed)
# --------------------------------------------------------------------------- #

def _fine_pixel_chi2(run_dir, regions, ny, nx):
    """Combine the per-region chi2 maps into one per-FINE-pixel map
    (max over regions): fine-grid regions map 1:1; coarse regions are
    looked up at the nearest observed pixel through the stored geometry
    (first-order warp inverse w ~ k - ds(k))."""
    total = np.zeros((ny, nx))
    for reg in regions:
        fobs = os.path.join(run_dir, 'out_' + reg['obs_file'])
        if not os.path.isfile(fobs):
            continue
        cmap, act = _region_chi2_map(run_dir, reg['obs_file'])
        cmap = np.where(act, cmap, 0.0)   # masked obs pixels: no constraint
        o = tiles.CoupledObs(os.path.join(run_dir, reg['obs_file']))
        if int(o.lts[0]) < 0:
            total = np.maximum(total, cmap)
        else:
            a = o.ltargs[:12]
            ky, kx = np.mgrid[0:ny, 0:nx].astype('float64')
            wx = kx - o.ds[0]
            wy = ky - o.ds[1]
            col = np.clip(np.round((wx + 0.5 - a[4]) / a[2]).astype(int),
                          0, int(a[0]) - 1)
            row = np.clip(np.round((wy + 0.5 - a[5]) / a[3]).astype(int),
                          0, int(a[1]) - 1)
            total = np.maximum(total, cmap[row, col])
    return total


def harvest_rundir(run_dir, apron, out_file=None, atmos_name=None):
    """Harvest one label record from a full-FOV coupled run dir.

    Works on any run whose output atmosphere + degraded synthetics
    (out_obs_*.nc) exist — including a partially converged inversion:
    the record carries a per-fine-pixel chi2 map, and training filters
    pixels by PIXEL_CHI2_MAX, so only well-fitted pixels become labels.
    """
    from . import stic_io
    with open(os.path.join(run_dir, 'input.cfg')) as f:
        cfg_text = f.read()
    regions = tiles.parse_regions(cfg_text)
    atmos_name = atmos_name or tiles.read_cfg_key(cfg_text, 'output_atmos')
    m = stic_io.Model.read(os.path.join(run_dir, atmos_name))
    ny, nx = m.ny, m.nx
    a = int(apron)
    sl = np.s_[a:ny - a, a:nx - a]

    rec_out = dict(
        src=run_dir, my0=0, mx0=0, apron=a,
        y_coords=np.arange(a, ny - a), x_coords=np.arange(a, nx - a),
        ltau=m.ltau[0, 0, 0],
        temp=m.temp[0][sl], vlos=m.vlos[0][sl], vturb=m.vturb[0][sl],
        blong=m.Bln[0][sl], bhor=m.Bho[0][sl], azi=m.azi[0][sl],
        pixel_chi2=_fine_pixel_chi2(run_dir, regions, ny, nx)[sl])

    for reg in regions:
        tob = tiles.CoupledObs(os.path.join(run_dir, reg['obs_file']))
        tag = os.path.splitext(reg['obs_file'])[0]
        rec_out[f'{tag}_dat'] = tob.dat[0].astype('float32')
        rec_out[f'{tag}_pweights'] = tob.pweights[0].astype('float32')
        rec_out[f'{tag}_wav'] = tob.wav
        rec_out[f'{tag}_weights'] = tob.weights.astype('float32')
        rec_out[f'{tag}_lts'] = tob.lts
        rec_out[f'{tag}_ltargs'] = tob.ltargs
        rec_out[f'{tag}_ds'] = tob.ds.astype('float32')
        if os.path.isfile(os.path.join(run_dir, 'out_' + reg['obs_file'])):
            c2, _ = _region_chi2(run_dir, reg['obs_file'])
            rec_out[f'{tag}_chi2'] = c2

    if out_file is None:
        os.makedirs(os.path.join(run_dir, 'labels'), exist_ok=True)
        out_file = os.path.join(run_dir, 'labels', 'full_map.npz')
    np.savez_compressed(out_file, **rec_out)
    pc = rec_out['pixel_chi2']
    print(f'harvest_rundir: {out_file}  labels {ny - 2 * a}x{nx - 2 * a}, '
          f'pixel chi2 median={np.nanmedian(pc):.1f} p90='
          f'{np.nanpercentile(pc, 90):.1f}')
    return out_file
