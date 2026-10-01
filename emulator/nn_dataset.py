"""
Turn harvested tile records (labels/tile_*.npz) into training samples,
and full run dirs into prediction inputs.

A sample (one tile) is a dict:
  regions: {tag: {'img':   (C_r+1, ny, nx) float32   # used channels + pweight
                  'rd':    (rdx, rdy)                # obs cell size, fine units
                  'rs':    (rsx, rsy)                # obs grid origin, absolute
                  'ds':    (2, tny, tnx) or None     # destretch maps
                  'm0':    (mx0, my0)}}              # tile origin, fine grid
  coords:  (npix, 2) absolute fine-grid (y, x) of the label pixels
  Y:       (npix, n_out) standardized targets        (training only)

The geometry lets the network map any absolute fine coordinate to a
continuous position in each region's image (see net.region_grid_coords).
"""

import os
import glob
import json
import hashlib
import numpy as np

from . import tiles
from . import emu_config as C


# --------------------------------------------------------------------------- #
# target encoding (same conventions as STIC_vers_2025/nn_emulator)
# --------------------------------------------------------------------------- #

def encode_targets_arrays(ltau, arrs):
    """arrs: dict var -> (ny, nx, ndep). Returns (Y_raw (npix, nout), names)."""
    ny, nx, nd = arrs['temp'].shape
    npix = ny * nx
    cols, names = [], []
    for v in C.TARGET_VARS:
        raw = arrs[v].reshape(npix, nd)
        grid = np.empty((npix, C.LTAU_GRID.size))
        for i in range(npix):
            grid[i] = np.interp(C.LTAU_GRID, ltau, raw[i])
        if v == 'temp' and C.LOG_TEMP:
            grid = np.log10(np.maximum(grid, 1.0))
            names += ['log_temp'] * grid.shape[1]
        elif v == 'azi' and C.AZI_SINCOS:
            s, c = np.sin(2 * grid), np.cos(2 * grid)
            grid = np.hstack([s, c])
            names += ['azi_sin'] * s.shape[1] + ['azi_cos'] * c.shape[1]
        else:
            names += [v] * grid.shape[1]
        cols.append(grid)
    return np.hstack(cols), names


def decode_targets(Y, names):
    names = np.asarray(names)
    out = {}
    for v in C.TARGET_VARS:
        if v == 'temp' and C.LOG_TEMP:
            out[v] = 10.0 ** Y[:, names == 'log_temp']
        elif v == 'azi' and C.AZI_SINCOS:
            out[v] = np.mod(0.5 * np.arctan2(Y[:, names == 'azi_sin'],
                                             Y[:, names == 'azi_cos']), np.pi)
        else:
            out[v] = Y[:, names == v]
    return out


# --------------------------------------------------------------------------- #
# region geometry + channel selection
# --------------------------------------------------------------------------- #

def _region_geometry(lts, ltargs, m0x, m0y):
    """(rdx, rdy, rsx, rsy) in absolute fine units for any region."""
    if int(np.ravel(lts)[0]) == 10:
        a = np.ravel(ltargs)
        return float(a[2]), float(a[3]), float(a[4]), float(a[5])
    # fine-grid region: obs pixel r center at absolute fine coord r + m0
    return 1.0, 1.0, float(m0x), float(m0y)


def _used_channels(weights):
    """Indices (w, s) of channels that carry data."""
    w, s = np.where(weights < 1e10)
    return w, s


def split_codes(src, coords, frac=None, block=None, guard=None, seed=None):
    """Spatial train/validation split of label pixels.

    Returns int8 codes per pixel: 0 train, 1 held out (validation),
    2 guard (training pixel next to a held-out block; used for neither).
    The held-out blocks are a deterministic function of (FOV, block
    index, seed) — SHA-256, not Python's per-process salted hash() (and
    not crc32, whose values cluster for such similar keys) — so all
    records of a FOV, all ensemble members and later evaluation scripts
    see the same split.
    """
    frac = C.VAL_FRACTION if frac is None else frac
    block = C.VAL_BLOCK if block is None else block
    guard = C.VAL_GUARD if guard is None else guard
    seed = C.VAL_SEED if seed is None else seed
    n = len(coords)
    if not frac or n == 0:
        return np.zeros(n, 'int8')
    y = np.asarray(coords[:, 0]).astype(int)
    x = np.asarray(coords[:, 1]).astype(int)
    cache = {}

    def held_out(by, bx):
        out = np.empty(by.size, bool)
        for i, key in enumerate(zip(by.tolist(), bx.tolist())):
            if key not in cache:
                h = hashlib.sha256(
                    f'{src}|{seed}|{key[0]}|{key[1]}'.encode()).digest()
                cache[key] = int.from_bytes(h[:8], 'big') / 2.0 ** 64 < frac
            out[i] = cache[key]
        return out

    val = held_out(y // block, x // block)
    near = np.zeros(n, bool)
    if guard > 0:
        # guard < block: the square of half-width `guard` around a pixel
        # touches at most the blocks of its four corners
        for dy in (-guard, guard):
            for dx in (-guard, guard):
                near |= held_out((y + dy) // block, (x + dx) // block)
    return np.where(val, 1, np.where(near, 2, 0)).astype('int8')


def record_to_sample(rec, stats=None, with_targets=True, chi2_thr=None):
    """One harvested npz record -> sample dict (see module docstring).

    chi2_thr: drop label pixels whose pixel_chi2 exceeds it (records
    without pixel_chi2 are not filtered).
    """
    tags = sorted({k.rsplit('_', 1)[0] for k in rec.files
                   if k.endswith('_dat')})
    m0x, m0y = int(rec['mx0']), int(rec['my0'])
    regions = {}
    for tag in tags:
        dat = np.asarray(rec[f'{tag}_dat'], 'float32')     # (ny, nx, nw, ns)
        wgt = np.asarray(rec[f'{tag}_weights'])
        pw = np.asarray(rec[f'{tag}_pweights'], 'float32')
        w, s = _used_channels(wgt)
        img = np.moveaxis(dat[:, :, w, s], -1, 0)          # (C, ny, nx)
        if stats is not None:
            mu, sd = stats[tag]
            img = (img - mu[:, None, None]) / sd[:, None, None]
        img = np.concatenate([np.nan_to_num(img), pw[None]], axis=0)
        rdx, rdy, rsx, rsy = _region_geometry(rec[f'{tag}_lts'],
                                              rec[f'{tag}_ltargs'], m0x, m0y)
        ds = np.asarray(rec[f'{tag}_ds'], 'float64') \
            if f'{tag}_ds' in rec.files and rec[f'{tag}_ds'].size else None
        regions[tag] = dict(img=img, rd=(rdx, rdy), rs=(rsx, rsy),
                            ds=ds, m0=(m0x, m0y))

    out = dict(regions=regions)
    if with_targets:
        yy, xx = np.asarray(rec['y_coords']), np.asarray(rec['x_coords'])
        gy, gx = np.meshgrid(yy, xx, indexing='ij')
        coords = np.stack([gy.ravel(), gx.ravel()], axis=1).astype('float64')
        arrs = {v: np.asarray(rec[v], 'float64') for v in C.TARGET_VARS}
        Y, names = encode_targets_arrays(np.asarray(rec['ltau'], 'float64'),
                                         arrs)
        good = np.isfinite(Y).all(axis=1)
        if 'pixel_chi2' in rec.files and chi2_thr is not None:
            good &= np.ravel(rec['pixel_chi2']) <= chi2_thr
        coords, Y = coords[good], Y[good]
        out['n_label_px'] = int(good.size)
        out['coords'] = coords
        out['Y_raw'] = Y
        out['target_names'] = names
        out['src'] = os.path.normpath(str(rec['src']))
    return out


def chi2_threshold(recs):
    """Global pixel-chi2 threshold over all records that carry pixel_chi2
    (full-map and tile records alike). Returns (threshold or None, how)."""
    if C.PIXEL_CHI2_MAX is not None:
        return float(C.PIXEL_CHI2_MAX), 'PIXEL_CHI2_MAX'
    if C.PIXEL_CHI2_QUANTILE is None:
        return None, 'off'
    pcs = [np.ravel(z['pixel_chi2']) for z in recs if 'pixel_chi2' in z.files]
    if not pcs:
        return None, 'no pixel_chi2 in records'
    pc = np.concatenate(pcs)
    pc = pc[np.isfinite(pc)]
    q = np.percentile(pc, [10, 50, 90])
    print(f'chi2_threshold: pixel chi2 over {pc.size} label pixels '
          f'p10/p50/p90 = {q[0]:.3g}/{q[1]:.3g}/{q[2]:.3g}')
    return (float(np.quantile(pc, C.PIXEL_CHI2_QUANTILE)),
            f'PIXEL_CHI2_QUANTILE={C.PIXEL_CHI2_QUANTILE}')


# --------------------------------------------------------------------------- #
# dataset assembly
# --------------------------------------------------------------------------- #

def load_records(label_dirs, chi2_max=None):
    files = []
    for d in label_dirs:
        found = sorted(glob.glob(os.path.join(d, '*.npz')))
        if not found:
            raise FileNotFoundError(f'no label records (*.npz) in {d}')
        files += found
    recs = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        if chi2_max is not None:
            c2 = [float(z[k]) for k in z.files if k.endswith('_chi2')]
            if any(np.isfinite(c) and c > chi2_max for c in c2):
                continue
        recs.append(z)
    print(f'load_records: {len(recs)}/{len(files)} tiles kept')
    return recs


def channel_stats(recs):
    """Per-region per-channel mean/std over all tiles' active pixels."""
    stats = {}
    tags = sorted({k.rsplit('_', 1)[0] for k in recs[0].files
                   if k.endswith('_dat')})
    for tag in tags:
        acc = []
        for z in recs:
            dat = np.asarray(z[f'{tag}_dat'], 'float64')
            w, s = _used_channels(np.asarray(z[f'{tag}_weights']))
            x = dat[:, :, w, s].reshape(-1, w.size)
            acc.append(x[np.isfinite(x).all(axis=1)])
        X = np.vstack(acc)
        mu = X.mean(0)
        sd = X.std(0)
        sd[sd < 1e-12] = 1.0
        stats[tag] = (mu.astype('float32'), sd.astype('float32'))
    return stats


def build_dataset(label_dirs, chi2_max=None):
    """Returns (samples, meta): standardized samples with standardized
    targets, plus everything needed to reproduce the encoding."""
    recs = load_records(label_dirs, chi2_max)
    stats = channel_stats(recs)
    thr, how = chi2_threshold(recs)
    raw = [record_to_sample(z, stats=stats, chi2_thr=thr) for z in recs]
    n_all = sum(s['n_label_px'] for s in raw)
    n_kept = sum(len(s['coords']) for s in raw)
    thr_txt = 'none' if thr is None else f'{thr:.4g}'
    print(f'build_dataset: pixel chi2 filter (<= {thr_txt}, {how}) kept '
          f'{n_kept}/{n_all} label pixels in {len(raw)} records')
    # a record whose pixels were all filtered out would give NaN losses
    # (mean over an empty batch) and silently untrained networks
    empty = [i for i, s in enumerate(raw) if len(s['coords']) == 0]
    if empty:
        print(f'build_dataset: dropping {len(empty)} record(s) with no '
              f'label pixels left after the chi2 filter')
    raw = [s for s in raw if len(s['coords']) > 0]
    if not raw:
        raise ValueError(
            'no label pixels left: the pixel chi2 filter removed all of '
            'them. Use the relative filter (emu_config.PIXEL_CHI2_MAX = '
            'None, PIXEL_CHI2_QUANTILE e.g. 0.5) or set PIXEL_CHI2_MAX '
            'from the p10/p50/p90 printed above.')
    for s in raw:
        s['split'] = split_codes(s['src'], s['coords'])
    # target standardization from TRAINING pixels only
    Ytr = np.vstack([s['Y_raw'][s['split'] == 0] for s in raw])
    if len(Ytr) == 0:
        Ytr = np.vstack([s['Y_raw'] for s in raw])
    ym, ys = Ytr.mean(0), Ytr.std(0)
    ys[ys < 1e-12] = 1.0
    for s in raw:
        s['Y'] = ((s['Y_raw'] - ym) / ys).astype('float32')
        del s['Y_raw']
    meta = dict(stats=stats, y_mean=ym, y_std=ys,
                target_names=raw[0]['target_names'],
                n_out=Ytr.shape[1],
                enc_channels={t: raw[0]['regions'][t]['img'].shape[0]
                              for t in raw[0]['regions']},
                chi2_threshold=thr, chi2_how=how,
                split=dict(frac=C.VAL_FRACTION, block=C.VAL_BLOCK,
                           guard=C.VAL_GUARD, seed=C.VAL_SEED))
    return raw, meta


def rundir_to_sample(run_dir, stats):
    """Full run dir -> prediction sample (all regions, full windows)."""
    with open(os.path.join(run_dir, 'input.cfg')) as f:
        regs = tiles.parse_regions(f.read())
    regions = {}
    for reg in regs:
        o = tiles.CoupledObs(os.path.join(run_dir, reg['obs_file']))
        tag = os.path.splitext(os.path.basename(reg['obs_file']))[0]
        w, s = _used_channels(o.weights)
        img = np.moveaxis(o.dat[0][:, :, w, s], -1, 0).astype('float32')
        mu, sd = stats[tag]
        img = (np.nan_to_num(img) - mu[:, None, None]) / sd[:, None, None]
        img = np.concatenate([img, o.pweights[0][None].astype('float32')], 0)
        rdx, rdy, rsx, rsy = _region_geometry(o.lts, o.ltargs, 0, 0)
        ds = o.ds.astype('float64') if o.ds.size else None
        regions[tag] = dict(img=img, rd=(rdx, rdy), rs=(rsx, rsy),
                            ds=ds, m0=(0, 0))
    fine = [t for t in regions if regions[t]['rd'] == (1.0, 1.0)]
    ny, nx = regions[fine[0]]['img'].shape[1:]
    return dict(regions=regions), (ny, nx)
