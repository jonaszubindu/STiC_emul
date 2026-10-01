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
            # epistemic std of this variable at the nearest grid depth
            gi = int(np.argmin(np.abs(grid - lt)))
            key = {'temp': 'log_temp'}.get(v, v)
            chans = [i for i, n in enumerate(names) if n == key]
            e = None
            if chans:
                ech = epi[sl][..., chans[gi]].reshape(npix)
                e = ech if v == 'temp' else ech * sc   # temp: dex
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
                    f'{row["nn_std"]:.3f}dex' if v == 'temp'
                    else f'{row["nn_std"]:.3g}')
                print(f'{lab + " [" + unit + "]":10s} {c:9s} {row["N"]:6d} '
                      f'{row["bias"]:9.3g} {row["mad"]:9.3g} '
                      f'{row["rms"]:9.3g} {row["corr"]:6.3f} {es:>9s}')

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'\nwrote {os.path.join(out_dir, "summary.json")}')
    _figures(maps, codes.reshape(shape), ltaus, out_dir)
    return summary


def _figures(maps, cat_map, ltaus, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print('matplotlib not installed: skipping figures '
              '(pip install matplotlib)')
        return
    show = [('temp', 'T [K]'), ('vlos', 'v_los [km/s]'),
            ('blong', 'B_long [G]')]
    for lt in ltaus:
        fig, axs = plt.subplots(len(show), 5, figsize=(17, 3.4 * len(show)),
                                constrained_layout=True)
        for r, (v, lab) in enumerate(show):
            inv, pred, d, e = maps[(lt, v)]
            lo, hi = np.nanpercentile(inv, [1, 99])
            dl = np.nanpercentile(np.abs(d), 99)
            panels = [(inv, f'inversion {lab}', 'viridis', lo, hi),
                      (pred, 'emulator', 'viridis', lo, hi),
                      (d, 'emulator - inversion', 'RdBu_r', -dl, dl)]
            for c, (img, title, cm, vmin, vmax) in enumerate(panels):
                im = axs[r, c].imshow(img, origin='lower', cmap=cm,
                                      vmin=vmin, vmax=vmax)
                axs[r, c].set_title(title)
                fig.colorbar(im, ax=axs[r, c], shrink=0.8)
            if e is not None:
                im = axs[r, 3].imshow(e, origin='lower', cmap='magma')
                axs[r, 3].set_title('ensemble std' +
                                    (' [dex]' if v == 'temp' else ''))
                fig.colorbar(im, ax=axs[r, 3], shrink=0.8)
            im = axs[r, 4].imshow(cat_map, origin='lower',
                                  cmap=ListedColormap(['#4c9be8', '#f2b134',
                                                       '#bbbbbb', '#d9534f']),
                                  vmin=-0.5, vmax=3.5)
            axs[r, 4].set_title('train / held-out / guard / dropped')
            cb = fig.colorbar(im, ax=axs[r, 4], ticks=[0, 1, 2, 3],
                              shrink=0.8)
            cb.ax.set_yticklabels(CATS)
        fig.suptitle(f'log tau = {lt:+.1f}')
        p = os.path.join(out_dir, f'compare_ltau{lt:+.1f}.png')
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print(f'wrote {p}')


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
