"""
Compare an emulator prediction against the inversion it was trained on.

Label pixels of a full-map record are split into the categories the
training saw:
  train     chi2-kept, used for training
  held-out  chi2-kept, in a held-out validation block (never trained on)
  guard     chi2-kept, next to a held-out block (used for neither)
  dropped   failed the pixel chi2 filter (no trustworthy reference)
For each category, variable and requested log(tau) the script prints
bias, median absolute difference, RMS, correlation and the ensemble's
median epistemic std, writes summary.json, and (if matplotlib is
available) one figure per log(tau).

Usage (GPU cluster, after predict_map):
  python -m emulator.compare_maps --labels <labels_map>/full_map.npz \
      --pred <round>/pred --ckpt <round>/ckpts --out <round>/compare
"""

import os
import json
import argparse
import numpy as np

from . import stic_io
from . import nn_dataset as nd

VARS = [  # name in record / model, label, scale to display units, unit
    ('temp', 'T', 1.0, 'K'),
    ('vlos', 'v_los', 1e-5, 'km/s'),
    ('vturb', 'v_turb', 1e-5, 'km/s'),
    ('blong', 'B_long', 1.0, 'G'),
    ('bhor', 'B_hor', 1.0, 'G'),
    ('azi', 'azimuth', 180.0 / np.pi, 'deg'),
]
_MODEL_ATTR = {'blong': 'Bln', 'bhor': 'Bho'}
CATS = ['train', 'held-out', 'guard', 'dropped']


def _at_ltau(ltau, cube, target):
    """cube (npix, ndep) on per-pixel-identical ltau -> (npix,) at target."""
    out = np.empty(cube.shape[0])
    for i in range(cube.shape[0]):
        out[i] = np.interp(target, ltau, cube[i])
    return out


def _categories(z, meta):
    """Category code per label pixel (0 train, 1 held-out, 2 guard,
    3 dropped) + the chi2 threshold used."""
    yy, xx = np.asarray(z['y_coords']), np.asarray(z['x_coords'])
    gy, gx = np.meshgrid(yy, xx, indexing='ij')
    coords = np.stack([gy.ravel(), gx.ravel()], 1).astype('float64')

    thr = meta.get('chi2_threshold')
    if thr is None and 'pixel_chi2' in z.files:
        # checkpoints from before the threshold was saved: recompute with
        # the current config (exact for single-record trainings)
        thr, _ = nd.chi2_threshold([z])
    kept = np.ones(len(coords), bool)
    if thr is not None and 'pixel_chi2' in z.files:
        kept = np.ravel(z['pixel_chi2']) <= thr

    split = meta.get('split')
    if split:
        codes = nd.split_codes(os.path.normpath(str(z['src'])), coords,
                               frac=split['frac'], block=split['block'],
                               guard=split['guard'], seed=split['seed'])
    else:
        codes = np.zeros(len(coords), 'int8')   # trained before hold-out
    codes = codes.astype(int)
    codes[~kept] = 3
    return codes, thr, (len(yy), len(xx)), (yy, xx)


def compare(labels_file, pred_dir, ckpt_dir, out_dir, ltaus):
    os.makedirs(out_dir, exist_ok=True)
    z = np.load(labels_file, allow_pickle=False)
    with open(os.path.join(ckpt_dir, 'meta.json')) as f:
        meta = json.load(f)
    codes, thr, shape, (yy, xx) = _categories(z, meta)

    m = stic_io.Model.read(os.path.join(pred_dir, 'predicted_atmos.nc'))
    pz = np.load(os.path.join(pred_dir, 'prediction.npz'))
    names = [str(s) for s in pz['target_names']]
    grid = np.asarray(pz['ltau_grid'])
    epi = pz['epistemic_std']                      # (ny, nx, n_out)

    sl = np.s_[yy[0]:yy[-1] + 1, xx[0]:xx[-1] + 1]
    lt_inv = np.asarray(z['ltau'], 'float64')
    lt_pred = m.ltau[0, 0, 0].astype('float64')
    npix = shape[0] * shape[1]

    print(f'label pixels {npix}: ' + ', '.join(
        f'{c} {int((codes == i).sum())}' for i, c in enumerate(CATS)))
    print(f'pixel chi2 threshold: {thr}')

    summary = dict(labels=labels_file, pred=pred_dir, ckpt=ckpt_dir,
                   chi2_threshold=thr,
                   counts={c: int((codes == i).sum())
                           for i, c in enumerate(CATS)},
                   metrics={})
    maps = {}
    spread_dex = {}
    for lt in ltaus:
        print(f'\n=== log tau = {lt:+.1f} ===')
        print(f'{"variable":10s} {"category":9s} {"N":>6s} {"bias":>9s} '
              f'{"MAD":>9s} {"RMS":>9s} {"corr":>6s} {"NN std":>9s}')
        for v, lab, sc, unit in VARS:
            inv = _at_ltau(lt_inv, np.asarray(z[v], 'float64')
                           .reshape(npix, -1), lt) * sc
            pr_cube = getattr(m, _MODEL_ATTR.get(v, v))[0][sl]
            pred = _at_ltau(lt_pred, pr_cube.reshape(npix, -1)
                            .astype('float64'), lt) * sc
            d = pred - inv
            if v == 'azi':                     # 180-degree ambiguity
                d = (d + 90.0) % 180.0 - 90.0
            # ensemble spread at this depth: in physical units from newer
            # predictions (std_<var> on the output depth grid), else from
            # the per-channel spread of older ones (temperature in dex)
            e, e_dex = None, False
            if f'std_{v}' in pz.files:
                e = _at_ltau(lt_pred, pz[f'std_{v}'][sl].reshape(npix, -1)
                             .astype('float64'), lt) * sc
            else:
                gi = int(np.argmin(np.abs(grid - lt)))
                key = {'temp': 'log_temp'}.get(v, v)
                chans = [i for i, n in enumerate(names) if n == key]
                if chans:
                    ech = epi[sl][..., chans[gi]].reshape(npix)
                    e, e_dex = (ech, True) if v == 'temp' else (ech * sc, False)
            spread_dex[v] = e_dex
            maps[(lt, v)] = (inv.reshape(shape), pred.reshape(shape),
                             d.reshape(shape),
                             None if e is None else e.reshape(shape))
            for i, c in enumerate(CATS):
                sel = codes == i
                if sel.sum() < 2:
                    continue
                corr = (np.corrcoef(inv[sel], pred[sel])[0, 1]
                        if v != 'azi' else np.nan)
                row = dict(N=int(sel.sum()),
                           bias=float(np.median(d[sel])),
                           mad=float(np.median(np.abs(d[sel]))),
                           rms=float(np.sqrt(np.mean(d[sel] ** 2))),
                           corr=float(corr),
                           nn_std=None if e is None
                           else float(np.median(e[sel])))
                summary['metrics'][f'{lt:+.1f}|{v}|{c}'] = row
                es = '' if e is None else (
                    f'{row["nn_std"]:.3f}dex' if e_dex
                    else f'{row["nn_std"]:.3g}')
                print(f'{lab + " [" + unit + "]":10s} {c:9s} {row["N"]:6d} '
                      f'{row["bias"]:9.3g} {row["mad"]:9.3g} '
                      f'{row["rms"]:9.3g} {row["corr"]:6.3f} {es:>9s}')

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'\nwrote {os.path.join(out_dir, "summary.json")}')
    _plots(maps, codes.reshape(shape), ltaus, out_dir,
           dict(z=z, m=m, sl=sl, epi=epi, names=names, grid=grid,
                shape=shape, pz=pz, spread_dex=spread_dex))
    return summary


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

# variable -> (label, colormap, symmetric about zero)
_STYLE = {'temp': ('T [K]', 'inferno', False),
          'vlos': ('v_los [km/s]', 'RdBu_r', True),
          'vturb': ('v_turb [km/s]', 'viridis', False),
          'blong': ('B_long [G]', 'RdGy_r', True),
          'bhor': ('B_hor [G]', 'viridis', False)}
_CAT_COLORS = ['#3b78c2', '#e8a33d', '#9a9a9a', '#c9473c']


def _plots(maps, cat_map, ltaus, out_dir, ctx):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed: skipping figures '
              '(pip install matplotlib)')
        return
    for lt in ltaus:
        _maps_figure(plt, maps, cat_map, lt, out_dir, ctx['spread_dex'])
        _scatter_figure(plt, maps, cat_map, lt, out_dir)
    _profiles_figure(plt, maps, cat_map, ltaus, out_dir, ctx)


def _save(plt, fig, out_dir, name):
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f'wrote {p}')


def _maps_figure(plt, maps, cat_map, lt, out_dir, spread_dex):
    """Inversion | emulator | difference | ensemble spread, one row per
    quantity. Solid outline: pixels excluded by the chi2 filter; dashed
    outline: held-out validation blocks."""
    rows = ['temp', 'vlos', 'blong', 'bhor']
    fig, axs = plt.subplots(len(rows), 4, figsize=(17, 3.6 * len(rows)),
                            constrained_layout=True, sharex=True, sharey=True)
    dropped = (cat_map == 3).astype(float)
    heldout = (cat_map == 1).astype(float)
    for r, v in enumerate(rows):
        lab, cmap, sym = _STYLE[v]
        inv, pred, d, e = maps[(lt, v)]
        if sym:
            hi = np.nanpercentile(np.abs(inv), 99)
            lo = -hi
        else:
            lo, hi = np.nanpercentile(inv, [1, 99])
        dl = np.nanpercentile(np.abs(d), 99)
        line = 'k' if sym else 'w'
        for c, (img, title, cm, vmin, vmax) in enumerate([
                (inv, f'inversion  {lab}', cmap, lo, hi),
                (pred, f'emulator  {lab}', cmap, lo, hi),
                (d, 'emulator - inversion', 'RdBu_r', -dl, dl)]):
            ax = axs[r, c]
            im = ax.imshow(img, origin='lower', cmap=cm, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=0.85)
            ax.set_title(title)
            if dropped.any() and c < 2:
                ax.contour(dropped, levels=[0.5], colors=line,
                           linewidths=0.6, alpha=0.7)
            if heldout.any() and c == 2:
                ax.contour(heldout, levels=[0.5], colors='k',
                           linewidths=0.9, linestyles='--')
        ax = axs[r, 3]
        if e is not None:
            im = ax.imshow(e, origin='lower', cmap='magma')
            fig.colorbar(im, ax=ax, shrink=0.85)
            ax.set_title('ensemble spread' + (' [dex]' if spread_dex.get(v)
                                              else f' [{lab.split("[")[1]}'))
        else:
            ax.set_axis_off()
    for ax in axs[-1]:
        ax.set_xlabel('x [px]')
    for ax in axs[:, 0]:
        ax.set_ylabel('y [px]')
    fig.suptitle(f'log tau = {lt:+.1f}    solid outline: excluded by the '
                 f'chi2 filter    dashed: held-out blocks', fontsize=12)
    _save(plt, fig, out_dir, f'maps_ltau{lt:+.1f}.png')


def _scatter_figure(plt, maps, cat_map, lt, out_dir):
    """Emulator vs inversion per pixel, coloured by category."""
    vars_ = ['temp', 'vlos', 'vturb', 'blong', 'bhor']
    cats = cat_map.ravel()
    fig, axs = plt.subplots(1, len(vars_), figsize=(4.2 * len(vars_), 4.4),
                            constrained_layout=True)
    for ax, v in zip(axs, vars_):
        lab = _STYLE[v][0]
        inv, pred = maps[(lt, v)][0].ravel(), maps[(lt, v)][1].ravel()
        lo, hi = np.nanpercentile(np.r_[inv, pred], [0.5, 99.5])
        stats = []
        # draw dropped first so trained / held-out pixels stay visible
        for i in (3, 2, 0, 1):
            sel = cats == i
            if not sel.any():
                continue
            ax.scatter(inv[sel], pred[sel], s=2, alpha=0.35,
                       color=_CAT_COLORS[i], label=CATS[i], rasterized=True)
        for i in (0, 1, 3):
            sel = cats == i
            if sel.sum() > 1:
                stats.append(f'{CATS[i]}: MAD {np.median(np.abs(pred[sel] - inv[sel])):.3g}')
        ax.plot([lo, hi], [lo, hi], 'k-', lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel(f'inversion  {lab}')
        ax.set_ylabel(f'emulator  {lab}')
        ax.text(0.03, 0.97, '\n'.join(stats), transform=ax.transAxes,
                va='top', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, lw=0))
    axs[0].legend(markerscale=5, loc='lower right', fontsize=8)
    fig.suptitle(f'log tau = {lt:+.1f}: emulator vs inversion per pixel '
                 f'(MAD = median absolute difference)')
    _save(plt, fig, out_dir, f'scatter_ltau{lt:+.1f}.png')


def _profiles_figure(plt, maps, cat_map, ltaus, out_dir, ctx):
    """Stratifications of characteristic pixels: inversion (black) vs
    emulator (red) with the ensemble spread as a band."""
    z, m, sl, epi = ctx['z'], ctx['m'], ctx['sl'], ctx['epi']
    names, grid, shape = ctx['names'], ctx['grid'], ctx['shape']
    npix = shape[0] * shape[1]
    cats = cat_map.ravel()
    lt_hi, lt_lo = max(ltaus), min(ltaus)          # deepest / highest
    t_deep = maps[(lt_hi, 'temp')][0].ravel()
    t_high = maps[(lt_lo, 'temp')][0].ravel()
    b_deep = maps[(lt_hi, 'blong')][0].ravel()

    def median_of(sel):
        idx = np.where(sel)[0]
        if idx.size == 0:
            return None
        return int(idx[np.argmin(np.abs(t_deep[idx] - np.median(t_deep[idx])))])

    picks = [(int(np.argmax(t_high)), f'hottest at log tau {lt_lo:+.0f}'),
             (int(np.argmin(t_deep)), f'coolest at log tau {lt_hi:+.0f}'),
             (int(np.argmax(np.abs(b_deep))), 'strongest |B_long|'),
             (median_of(cats == 0), 'typical training pixel'),
             (median_of(cats == 1), 'typical held-out pixel'),
             (median_of(cats == 3), 'typical chi2-excluded pixel')]
    seen, rows = set(), []
    for i, what in picks:
        if i is not None and i not in seen:
            seen.add(i)
            rows.append((i, what))

    lt_inv = np.asarray(z['ltau'], 'float64')
    lt_pred = m.ltau[0, 0, 0].astype('float64')
    # (variable, label, scale, minimum y-range so small wiggles stay small)
    cols = [('temp', 'T [K]', 1.0, None), ('vlos', 'v_los [km/s]', 1e-5, 1.0),
            ('blong', 'B_long [G]', 1.0, 50.0)]
    fig, axs = plt.subplots(len(rows), 3, figsize=(13, 2.6 * len(rows)),
                            constrained_layout=True, sharex=True)
    axs = np.atleast_2d(axs)
    for r, (i, what) in enumerate(rows):
        yy, xx = np.unravel_index(i, shape)
        for c, (v, lab, sc, min_span) in enumerate(cols):
            ax = axs[r, c]
            inv = np.asarray(z[v], 'float64').reshape(npix, -1)[i] * sc
            pr = getattr(m, _MODEL_ATTR.get(v, v))[0][sl] \
                .reshape(npix, -1)[i].astype('float64') * sc
            ax.plot(lt_inv, inv, 'k-', lw=1.4, label='inversion')
            ax.plot(lt_pred, pr, 'r-', lw=1.2, label='emulator')
            key = 'log_temp' if v == 'temp' else v
            chans = [k for k, n in enumerate(names) if n == key]
            pz = ctx['pz']
            if f'std_{v}' in pz.files:
                sig = pz[f'std_{v}'][sl].reshape(npix, -1)[i] * sc
                ax.fill_between(lt_pred, pr - sig, pr + sig, color='r',
                                alpha=0.2, lw=0, label='ensemble spread')
            elif chans:
                sig = epi[sl].reshape(npix, -1)[i, chans]
                mid = np.interp(grid, lt_pred, pr)
                if v == 'temp':
                    lo, hi = mid * 10 ** (-sig), mid * 10 ** sig
                else:
                    lo, hi = mid - sig * sc, mid + sig * sc
                ax.fill_between(grid, lo, hi, color='r', alpha=0.2, lw=0,
                                label='ensemble spread')
            ax.set_xlim(lt_pred.min(), lt_pred.max())
            if v == 'temp':
                ax.set_yscale('log')
            elif min_span is not None:
                y0, y1 = ax.get_ylim()
                if y1 - y0 < min_span:
                    mid = 0.5 * (y0 + y1)
                    ax.set_ylim(mid - 0.5 * min_span, mid + 0.5 * min_span)
            if c == 0:
                ax.set_ylabel(f'{what}\n(y={yy}, x={xx}, {CATS[cats[i]]})',
                              fontsize=8)
            if r == 0:
                ax.set_title(lab)
            if r == len(rows) - 1:
                ax.set_xlabel('log tau (500 nm)')
    axs[0, 0].legend(fontsize=8)
    fig.suptitle('Stratifications of characteristic pixels')
    _save(plt, fig, out_dir, 'profiles.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', required=True, help='full_map.npz record')
    ap.add_argument('--pred', required=True, help='predict_map output dir')
    ap.add_argument('--ckpt', required=True, help='training checkpoint dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ltau', type=float, nargs='+', default=[-1.0, -3.0, -5.0])
    a = ap.parse_args()
    compare(a.labels, a.pred, a.ckpt, a.out, a.ltau)
