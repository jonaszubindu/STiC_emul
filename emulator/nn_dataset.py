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
# target encoding
#
# Every output channel has a quantity name (e.g. 'log_temp') and a log tau
# position. Channels with the same name form a group, and a group's depth
# profile is the linear interpolation between its positions. With the
# inversion's own nodes as positions ('nodes' representation) this is
# exactly how STiC writes the atmosphere of a coupled (mode 5) inversion
# (depthmodel.cc expandAtmos -> expand -> linpol(..., extrapolate=true)):
# linear between nodes AND linearly continued beyond the outermost ones
# (a single node: constant). Fixed-grid groups are constant beyond.
# --------------------------------------------------------------------------- #

def interp_matrix(xx, x, extrapolate=False):
    """A with A @ y == np.interp(xx, x, y) for any y (x ascending).
    extrapolate: continue the first/last segment linearly beyond the
    outermost points instead of holding their values (STiC's linpol)."""
    xx = np.asarray(xx, 'float64')
    x = np.asarray(x, 'float64')
    A = np.zeros((xx.size, x.size))
    if x.size == 1:
        A[:, 0] = 1.0
        return A
    for i, t in enumerate(xx):
        if t <= x[0] and not extrapolate:
            A[i, 0] = 1.0
        elif t >= x[-1] and not extrapolate:
            A[i, -1] = 1.0
        else:
            # segment containing t (x[k-1] < t <= x[k]); the end segments
            # beyond the outermost points
            k = int(np.clip(np.searchsorted(x, t), 1, x.size - 1))
            w = (t - x[k - 1]) / (x[k] - x[k - 1])
            A[i, k - 1], A[i, k] = 1.0 - w, w
    return A


# output channel name -> atmospheric quantity it encodes
_CHANNEL_VAR = {'log_temp': 'temp', 'temp': 'temp', 'vlos': 'vlos',
                'vturb': 'vturb', 'blong': 'blong', 'bhor': 'bhor',
                'bperp_c': 'bhor', 'bperp_s': 'bhor',
                'azi': 'azi', 'azi_sin': 'azi', 'azi_cos': 'azi'}


def target_extrap(names, exact):
    """Per output channel: True where its group sits at STiC nodes (and is
    therefore continued linearly beyond the outermost node, like STiC)."""
    return np.array([bool(exact[_CHANNEL_VAR[str(n)]]) for n in names])


def target_positions(nodes=None):
    """log tau positions of the output channels per quantity: the
    inversion's nodes where given, otherwise LTAU_GRID (also for
    quantities that were not inverted)."""
    pos, exact = {}, {}
    for v in C.TARGET_VARS:
        n = None if nodes is None else nodes.get(v)
        exact[v] = n is not None and len(n) > 0
        pos[v] = np.sort(np.asarray(n, 'float64')) if exact[v] \
            else C.LTAU_GRID.astype('float64')
    if C.B_PERP_VECTOR:
        pos['bperp'] = np.union1d(pos['bhor'], pos['azi'])
    return pos, exact


def profile_to_targets(ltau, prof, pos, exact):
    """(npix, ndep) profiles -> (npix, npos) values at `pos`. exact: the
    profiles are piecewise linear on `pos` (STiC nodes) and the node
    values are recovered by least squares (also off-grid nodes);
    otherwise the profiles are sampled at `pos`."""
    if exact:
        A = interp_matrix(ltau, pos, extrapolate=True)   # depth <- pos
        sol = np.linalg.lstsq(A, np.asarray(prof, 'float64').T, rcond=None)[0]
        return sol.T
    return np.asarray(prof, 'float64') @ interp_matrix(pos, ltau).T


def encode_targets_arrays(ltau, arrs, pos, exact):
    """arrs: dict var -> (ny, nx, ndep). Returns (Y_raw (npix, nout),
    names, where) with `where` the log tau position of every channel."""
    ny, nx, nd = arrs['temp'].shape
    npix = ny * nx
    cols, names, where = [], [], []

    def add(name, vals, p):
        cols.append(vals)
        names.extend([name] * len(p))
        where.extend(list(p))

    def at(v):
        return profile_to_targets(ltau, arrs[v].reshape(npix, nd),
                                  pos[v], exact[v])

    for v in C.TARGET_VARS:
        if C.B_PERP_VECTOR and v == 'azi':
            continue                    # folded into the transverse vector
        if C.B_PERP_VECTOR and v == 'bhor':
            # B_hor and azimuth at their own nodes (as STiC builds them),
            # then the vector (B_hor cos 2phi, B_hor sin 2phi) at the union
            P = pos['bperp']
            bh = at('bhor') @ interp_matrix(P, pos['bhor'],
                                            exact['bhor']).T
            az = at('azi') @ interp_matrix(P, pos['azi'], exact['azi']).T
            add('bperp_c', bh * np.cos(2 * az), P)
            add('bperp_s', bh * np.sin(2 * az), P)
            continue
        vals = at(v)
        if v == 'temp' and C.LOG_TEMP:
            add('log_temp', np.log10(np.maximum(vals, 1.0)), pos[v])
        elif v == 'azi' and C.AZI_SINCOS:
            add('azi_sin', np.sin(2 * vals), pos[v])
            add('azi_cos', np.cos(2 * vals), pos[v])
        else:
            add(v, vals, pos[v])
    return np.hstack(cols), names, np.asarray(where, 'float64')


def stratify(Y, names, where, depth, extrap=None):
    """Outputs in target units -> physical stratifications on `depth`,
    built like STiC: linear between channel positions; beyond the
    outermost ones continued linearly where `extrap` (per channel, see
    target_extrap) is set, constant otherwise. Works for every encoding
    (old 16-point grid checkpoints included)."""
    names = np.asarray(names)
    where = np.asarray(where, 'float64')
    Y = np.asarray(Y, 'float64')
    extrap = np.zeros(names.size, bool) if extrap is None \
        else np.asarray(extrap, bool)

    def has(n):
        return bool(np.any(names == n))

    def grp(n, f=None):
        sel = np.where(names == n)[0]
        order = np.argsort(where[sel], kind='stable')
        x, v = where[sel][order], Y[:, sel[order]]
        if f is not None:
            v = f(v)
        return v @ interp_matrix(depth, x, bool(extrap[sel].any())).T

    out = {}
    out['temp'] = grp('log_temp', lambda v: 10.0 ** v) if has('log_temp') \
        else grp('temp')
    for v in ('vlos', 'vturb', 'blong'):
        if has(v):
            out[v] = grp(v)
    if has('bperp_c'):
        c, s = grp('bperp_c'), grp('bperp_s')
        out['bhor'] = np.hypot(c, s)
        out['azi'] = np.mod(0.5 * np.arctan2(s, c), np.pi)
    else:
        if has('bhor'):
            out['bhor'] = grp('bhor')
        if has('azi_sin'):
            out['azi'] = np.mod(0.5 * np.arctan2(grp('azi_sin'),
                                                 grp('azi_cos')), np.pi)
        elif has('azi'):
            out['azi'] = grp('azi')
    return out


def dataset_nodes(recs):
    """Node positions per quantity for the 'nodes' representation: stored
    in the records at harvest time, or computed from emu_config.NODES_CFG
    for records harvested before that. Must agree across records.

    A quantity without nodes (nodes_<var> = 0 in the cfg: not inverted in
    that cycle, so its profile comes from an earlier cycle or the starting
    model) gets its nodes inferred from the label profiles (infer_nodes)."""
    if C.TARGET_REPR != 'nodes':
        return None
    found = []
    for z in recs:
        if all(f'nodes_{v}' in z.files for v in C.TARGET_VARS):
            found.append({v: np.asarray(z[f'nodes_{v}'], 'float64')
                          for v in C.TARGET_VARS})
        elif C.NODES_CFG:
            found.append(tiles.stic_nodes_cycles(
                tiles.read_cfgs(C.NODES_CFG), z['ltau']))
        else:
            raise ValueError(
                'a label record carries no node positions: re-harvest it '
                'with the current code, or set emu_config.NODES_CFG to the '
                'input.cfg(s) of the inversion cycles that produced it')
    ref = found[0]
    for d in found[1:]:
        for v in C.TARGET_VARS:
            if d[v].shape != ref[v].shape or not np.allclose(d[v], ref[v]):
                raise ValueError(f'label records use different {v} nodes; '
                                 'one network needs one node setup')
    ltau = np.asarray(recs[0]['ltau'], 'float64')
    for v in C.TARGET_VARS:
        if len(ref[v]) == 0:
            ref[v] = infer_nodes(ltau, [z[v] for z in recs])
            print(f'dataset_nodes: {v} has no nodes in the cfg; '
                  f'{len(ref[v])} nodes inferred from the label profiles '
                  'at log tau ' + ' '.join(f'{x:.2f}' for x in ref[v]))
    return ref


def infer_nodes(ltau, profiles):
    """Node positions that reproduce piecewise-linear STiC profiles
    exactly, read off the profiles themselves. STiC snaps nodes to the
    depth grid and writes straight lines between them, linearly continued
    beyond the outermost ones, so a profile bends only at its nodes: the
    nodes are the grid points where the slope changes in at least
    INFER_NODES_MIN_FRAC of the pixels, plus both grid ends (a linear
    continuation beyond an end node is the same as a node at the grid end).
    Slope changes below float32 rounding of the stored labels
    (INFER_NODES_TOL machine epsilons) do not count.

    profiles: list of (..., ndep) arrays on the common grid `ltau`."""
    x = np.asarray(ltau, 'float64')
    order = np.argsort(x)
    x = x[order]
    dx = np.diff(x)
    hits = np.zeros(x.size, 'int64')
    n = 0
    for p in profiles:
        p = np.asarray(p, 'float64').reshape(-1, x.size)[:, order]
        p = p[np.isfinite(p).all(axis=1)]
        s = np.diff(p, axis=1) / dx                      # segment slopes
        bend = np.abs(np.diff(s, axis=1))                # at x[1:-1]
        scale = np.abs(p).max(axis=1, keepdims=True)
        tol = (C.INFER_NODES_TOL * np.finfo('float32').eps * scale
               * (1.0 / dx[:-1] + 1.0 / dx[1:]))
        hits[1:-1] += (bend > tol).sum(axis=0)
        n += len(p)
    inner = hits >= max(1, C.INFER_NODES_MIN_FRAC * n)
    inner[[0, -1]] = True
    return x[inner]


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


def region_channels(dat, weights, pol_over_i=False):
    """(ny, nx, nw, ns) profiles -> (C, ny, nx) float32 of the used
    channels. pol_over_i: Stokes Q, U, V are divided by the pixel's mean
    Stokes I over the used wavelengths of this region."""
    w, s = _used_channels(np.asarray(weights))
    x = np.asarray(dat, 'float64')[:, :, w, s]          # (ny, nx, C)
    if pol_over_i and (s == 0).any() and (s > 0).any():
        i_mean = x[..., s == 0].mean(axis=-1)
        pos = i_mean[np.isfinite(i_mean) & (i_mean > 0)]
        floor = 1e-3 * np.median(pos) if pos.size else 1.0
        x[..., s > 0] /= np.maximum(i_mean, floor)[..., None]
    return np.moveaxis(x, -1, 0).astype('float32')


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


def record_to_sample(rec, stats=None, with_targets=True, chi2_thr=None,
                     pos=None, exact=None, pol_over_i=False):
    """One harvested npz record -> sample dict (see module docstring).

    chi2_thr: drop label pixels whose pixel_chi2 exceeds it (records
    without pixel_chi2 are not filtered).
    """
    tags = sorted({k.rsplit('_', 1)[0] for k in rec.files
                   if k.endswith('_dat')})
    m0x, m0y = int(rec['mx0']), int(rec['my0'])
    regions = {}
    for tag in tags:
        pw = np.asarray(rec[f'{tag}_pweights'], 'float32')
        img = region_channels(rec[f'{tag}_dat'], rec[f'{tag}_weights'],
                              pol_over_i)                 # (C, ny, nx)
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
        if pos is None:
            pos, exact = target_positions(None)
        Y, names, where = encode_targets_arrays(
            np.asarray(rec['ltau'], 'float64'), arrs, pos, exact)
        good = np.isfinite(Y).all(axis=1)
        if 'pixel_chi2' in rec.files and chi2_thr is not None:
            good &= np.ravel(rec['pixel_chi2']) <= chi2_thr
        coords, Y = coords[good], Y[good]
        out['n_label_px'] = int(good.size)
        out['coords'] = coords
        out['Y_raw'] = Y
        out['target_names'] = names
        out['target_ltau'] = where
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


def channel_stats(recs, pol_over_i=False):
    """Per-region per-channel mean/std over all tiles' active pixels."""
    stats = {}
    tags = sorted({k.rsplit('_', 1)[0] for k in recs[0].files
                   if k.endswith('_dat')})
    for tag in tags:
        acc = []
        for z in recs:
            img = region_channels(z[f'{tag}_dat'], z[f'{tag}_weights'],
                                  pol_over_i)
            x = img.reshape(img.shape[0], -1).T.astype('float64')
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
    stats = channel_stats(recs, C.POL_OVER_I)
    thr, how = chi2_threshold(recs)
    nodes = dataset_nodes(recs)
    pos, exact = target_positions(nodes)
    depth = np.asarray(recs[0]['ltau'], 'float64')
    _check_representation(recs[0], pos, exact, depth)
    raw = [record_to_sample(z, stats=stats, chi2_thr=thr, pos=pos,
                            exact=exact, pol_over_i=C.POL_OVER_I)
           for z in recs]
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
    ys = _floor_spread(ys, raw[0]['target_names'])
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
                           guard=C.VAL_GUARD, seed=C.VAL_SEED),
                target_ltau=raw[0]['target_ltau'],
                target_extrap=target_extrap(raw[0]['target_names'], exact),
                depth_grid=depth,
                target_repr=C.TARGET_REPR if nodes is not None else 'grid',
                nodes={v: [float(x) for x in nodes[v]] for v in nodes}
                if nodes is not None else None,
                pol_over_i=bool(C.POL_OVER_I),
                b_perp_vector=bool(C.B_PERP_VECTOR),
                channels=input_channels(recs[0]))
    return raw, meta


def input_channels(rec):
    """Wavelength and Stokes index of every network input channel per
    region, so that new data can be checked against the training layout."""
    out = {}
    tags = sorted({k.rsplit('_', 1)[0] for k in rec.files
                   if k.endswith('_dat')})
    for tag in tags:
        w, s = _used_channels(np.asarray(rec[f'{tag}_weights']))
        out[tag] = (np.asarray(rec[f'{tag}_wav'], 'float64')[w],
                    s.astype('int32'))
    return out


def _floor_spread(ys, names):
    """Outputs that barely vary across the training pixels (e.g. a node
    the inversion left at its starting value) would be divided by ~0 and
    turn any small held-out difference into a huge standardized value.
    Their spread is floored at a physically negligible size
    (emu_config.TARGET_SPREAD_FLOOR)."""
    names = np.asarray(names)
    out = np.asarray(ys, 'float64').copy()
    floor = np.array([C.TARGET_SPREAD_FLOOR.get(str(n), 0.0) for n in names])
    out = np.maximum(out, floor)
    out[out < 1e-12] = 1.0
    n_fl = int((out > ys * (1 + 1e-9)).sum())
    if n_fl:
        print(f'build_dataset: {n_fl} output(s) barely vary across the '
              f'training pixels; standardized with the minimum spread')
    return out


def _check_representation(rec, pos, exact, depth):
    """How well the output representation reproduces the labels of one
    record: exact up to float32 precision when the nodes match the
    inversion. Above emu_config.REPR_TOL the network could not learn the
    labels even in principle (systematic offsets also on training pixels),
    so training stops unless REPR_CHECK = 'warn'."""
    arrs = {v: np.asarray(rec[v], 'float64') for v in C.TARGET_VARS}
    Y, names, where = encode_targets_arrays(depth, arrs, pos, exact)
    back = stratify(Y, names, where, depth, target_extrap(names, exact))
    msg, bad = [], []
    for v, unit, sc in (('temp', 'K', 1.0), ('vlos', 'km/s', 1e-5),
                        ('vturb', 'km/s', 1e-5), ('blong', 'G', 1.0),
                        ('bhor', 'G', 1.0)):
        lab = arrs[v].reshape(back[v].shape)
        err = np.abs(back[v] - lab)
        imax = np.unravel_index(np.argmax(err), err.shape)
        msg.append(f'{v} {err[imax] * sc:.3g} {unit}')
        if err[imax] > C.REPR_TOL[v]:
            bad.append(f'{v} (max {err[imax] * sc:.3g} {unit} at log tau '
                       f'{depth[imax[-1]]:.2f})')
    print('representation check (max |rebuilt - label| over one record): '
          + ', '.join(msg))
    for v in C.TARGET_VARS:
        how = (f'{len(pos[v])} STiC nodes' if exact[v]
               else f'{len(pos[v])}-point grid')
        print(f'  {v}: {how} at log tau ' +
              ' '.join(f'{x:.2f}' for x in pos[v]))
    if bad and C.TARGET_REPR == 'nodes':
        txt = ('the output representation does not reproduce the labels: '
               + '; '.join(bad) + '. The node positions (from the cfg, or '
               'inferred from the profiles) are not the ones that built '
               'these profiles. A quantity keeps the profile of the LAST '
               'cycle that inverted it (nodes_<var> != 0); giving the '
               'input.cfg of every cycle fixes the nodes: '
               'labels.restore_nodes(record, [cfg_cycle1, cfg_cycle2, '
               '...]) or emu_config.NODES_CFG.')
        if C.REPR_CHECK == 'warn':
            print('WARNING: ' + txt)
        else:
            raise ValueError(txt + " (emu_config.REPR_CHECK = 'warn' "
                             'trains anyway)')


def rundir_to_sample(run_dir, stats, pol_over_i=False):
    """Full run dir -> prediction sample (all regions, full windows)."""
    with open(os.path.join(run_dir, 'input.cfg')) as f:
        regs = tiles.parse_regions(f.read())
    regions = {}
    for reg in regs:
        tag = os.path.splitext(os.path.basename(reg['obs_file']))[0]
        if tag not in stats:
            continue                    # region the network was not trained on
        o = tiles.CoupledObs(os.path.join(run_dir, reg['obs_file']))
        img = region_channels(o.dat[0], o.weights, pol_over_i)
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
