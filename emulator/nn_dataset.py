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


def record_to_sample(rec, stats=None, with_targets=True):
    """One harvested npz record -> sample dict (see module docstring)."""
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
        # per-pixel chi2 filter (full-map records from harvest_rundir)
        if 'pixel_chi2' in rec.files and (C.PIXEL_CHI2_MAX is not None
                                          or C.PIXEL_CHI2_QUANTILE is not None):
            pc = np.ravel(rec['pixel_chi2'])
            if C.PIXEL_CHI2_MAX is not None:
                thr, how = C.PIXEL_CHI2_MAX, 'PIXEL_CHI2_MAX'
            else:
                thr = float(np.nanquantile(pc, C.PIXEL_CHI2_QUANTILE))
                how = f'PIXEL_CHI2_QUANTILE={C.PIXEL_CHI2_QUANTILE}'
            good = (pc <= thr) & np.isfinite(Y).all(axis=1)
            coords, Y = coords[good], Y[good]
            q = np.nanpercentile(pc, [10, 50, 90])
            print(f'record_to_sample: pixel chi2 filter (<= {thr:.4g}, '
                  f'{how}) kept {good.sum()}/{good.size} label pixels; '
                  f'pixel chi2 p10/p50/p90 = '
                  f'{q[0]:.3g}/{q[1]:.3g}/{q[2]:.3g}')
        out['coords'] = coords
        out['Y_raw'] = Y
        out['target_names'] = names
    return out


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
    raw = [record_to_sample(z, stats=stats) for z in recs]
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
    Yall = np.vstack([s['Y_raw'] for s in raw])
    ym, ys = Yall.mean(0), Yall.std(0)
    ys[ys < 1e-12] = 1.0
    for s in raw:
        s['Y'] = ((s['Y_raw'] - ym) / ys).astype('float32')
        del s['Y_raw']
    meta = dict(stats=stats, y_mean=ym, y_std=ys,
                target_names=raw[0]['target_names'],
                n_out=Yall.shape[1],
                enc_channels={t: raw[0]['regions'][t]['img'].shape[0]
                              for t in raw[0]['regions']})
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
