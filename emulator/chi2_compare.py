"""
Data-space check of an emulator prediction: does the emulator's atmosphere
fit the observations as well as the inversion's?

Both atmospheres are synthesized with STiC (mode 2), degraded to every
instrument grid exactly like coupled STiC does it (labels.degrade_synthetic)
and compared with the observations; the chi2 of both is then compared per
pixel and per spectral region.

Steps (paths are examples; <launch> is the launch dir with input.cfg):

 1. model: the emulator atmosphere with the inversion's gas-pressure
    boundary (STiC integrates hydrostatic equilibrium from the top gas
    pressure; the emulator does not predict it):
      python -m emulator.chi2_compare model --pred <pred>/predicted_atmos.nc \\
          --inv <launch>/atmosout.nc --out <chi2>/emulator_model.nc

 2. STiC mode 2 on <chi2>/emulator_model.nc -> synthetic profiles (and on
    the inversion's atmosphere, if its mode-2 synthetic is not at hand).

 3. degrade: both synthetics to the instrument grids, each into its own
    directory (never into <launch>, which holds the inversion's out_ files):
      python -m emulator.chi2_compare degrade --run <launch> \\
          --synth <syn_emulator.nc> --out <chi2>/emulator
      python -m emulator.chi2_compare degrade --run <launch> \\
          --synth <syn_inversion.nc> --out <chi2>/inversion

 4. compare:
      python -m emulator.chi2_compare compare --run <launch> \\
          --inv <chi2>/inversion --emu <chi2>/emulator --out <chi2>/compare \\
          [--labels <labels>/full_map.npz --ckpt <ckpts>]
    With --labels/--ckpt the pixels are split into the training categories
    (train / held-out / guard / dropped) like compare_maps does.
"""

import os
import json
import argparse
import numpy as np

from . import stic_io, tiles, labels
from .labels import region_tag

CATS = ['train', 'held-out', 'guard', 'dropped']
_CAT_COLORS = ['#3b78c2', '#e8a33d', '#9a9a9a', '#c9473c']


# --------------------------------------------------------------------------- #
# step 1: synthesis model
# --------------------------------------------------------------------------- #

def synthesis_model(pred_file, inv_file, out_file):
    """Emulator atmosphere + the inversion's pressure boundary (gas
    pressure, density, electron density, transition-region parameters) so
    that both atmospheres are synthesized under identical conditions."""
    p = stic_io.Model.read(pred_file)
    q = stic_io.Model.read(inv_file, t0=0, t1=1)
    if (p.ny, p.nx, p.ndep) != (q.ny, q.nx, q.ndep):
        raise ValueError(
            f'emulator atmosphere {p.ny}x{p.nx}x{p.ndep} and inversion '
            f'atmosphere {q.ny}x{q.nx}x{q.ndep} differ in shape. The '
            'prediction must cover the inversion\'s full FOV on its depth '
            'grid (checkpoints trained since the node targets do this).')
    if not np.allclose(p.ltau[0], q.ltau[0], atol=1e-4):
        raise ValueError('emulator and inversion use different log tau '
                         'grids')
    for a in ('pgas', 'rho', 'nne', 'tr_loc', 'tr_amp', 'tr_N'):
        getattr(p, a)[0] = getattr(q, a)[0]
    top = q.pgas[0][..., 0]
    if not np.all(top > 0):
        print('model: WARNING the inversion atmosphere has no positive top '
              'gas pressure everywhere; STiC then takes the boundary from '
              'rho or nne (copied as well)')
    p.write(out_file, write_all=True)
    print(f'model: wrote {out_file}  (top gas pressure from {inv_file}: '
          f'{np.nanmin(top):.3g} .. {np.nanmax(top):.3g})')
    return out_file


# --------------------------------------------------------------------------- #
# step 3: degradation
# --------------------------------------------------------------------------- #

def degrade(run_dir, synth_file, out_dir):
    if os.path.realpath(out_dir) == os.path.realpath(run_dir):
        raise ValueError('write the degraded synthetics into their own '
                         'directory, not the launch dir: it holds the '
                         "inversion's out_ files")
    os.makedirs(out_dir, exist_ok=True)
    return labels.degrade_synthetic(run_dir, synth_file, out_dir)


# --------------------------------------------------------------------------- #
# step 4: comparison
# --------------------------------------------------------------------------- #

def _regions(run_dir):
    with open(os.path.join(run_dir, 'input.cfg')) as f:
        regs = tiles.parse_regions(f.read())
    obs = {r['obs_file']: tiles.CoupledObs(os.path.join(run_dir, r['obs_file']))
           for r in regs}
    fine = [r for r in regs if int(obs[r['obs_file']].lts[0]) < 0]
    if not fine:
        raise ValueError('no region on the model grid (lts = -1) in '
                         'input.cfg; cannot define the fine pixel grid')
    o = obs[fine[0]['obs_file']]
    return regs, obs, (o.ny, o.nx)


def _read_out(run_dir, out_dir, obs_file):
    from netCDF4 import Dataset
    with Dataset(labels.out_obs_path(out_dir, obs_file)) as f:
        return np.ma.filled(f.variables['profiles'][0], np.nan)


def _region_parts(run_dir, out_dir, obs_file, o):
    """chi2 of one region against the observations on the region's grid,
    split into the Stokes I and the Q, U, V contributions (same
    normalization as STiC's chi2, so I + QUV = total). For a region with a
    single fitted wavelength (a continuum point) also the relative
    intensity error |obs - syn| / obs: there the residual passes close to
    zero for many pixels and a chi2 ratio is ill-conditioned."""
    syn = _read_out(run_dir, out_dir, obs_file)
    use = o.weights < 1e10                              # (nw, ns)
    sig = np.where(use, o.weights, np.inf)
    r = (o.dat[0] - syn) / sig[None, None]
    r2 = np.where(np.isfinite(r), r * r, 0.0)
    ndata = max(int(use.sum()), 1)
    act = o.pweights[0] > 0
    nan = lambda a: np.where(act, a, np.nan)
    out = dict(total=nan(r2.sum(axis=(2, 3)) / ndata),
               I=nan(r2[..., 0].sum(axis=-1) / ndata), QUV=None, rel=None)
    if o.ns > 1 and use[:, 1:].any():
        out['QUV'] = nan(r2[..., 1:].sum(axis=(2, 3)) / ndata)
    wl = np.where(use.any(axis=1))[0]
    if wl.size == 1 and use[wl[0], 0]:
        ob = o.dat[0, :, :, wl[0], 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            out['rel'] = nan(np.abs(ob - syn[:, :, wl[0], 0]) / np.abs(ob))
    return out


def _rel_stats(ri, re):
    ok = np.isfinite(ri) & np.isfinite(re)
    if ok.sum() == 0:
        return None
    return dict(N=int(ok.sum()),
                rel_err_inv_median=float(np.median(ri[ok])),
                rel_err_emu_median=float(np.median(re[ok])),
                rel_err_inv_p90=float(np.percentile(ri[ok], 90)),
                rel_err_emu_p90=float(np.percentile(re[ok], 90)),
                frac_emu_closer=float(np.mean(re[ok] <= ri[ok])))


def _category_map(labels_file, ckpt_dir, shape):
    """Training category per fine pixel (-1: outside the label record,
    e.g. the apron)."""
    from .compare_maps import _categories
    z = np.load(labels_file, allow_pickle=False)
    with open(os.path.join(ckpt_dir, 'meta.json')) as f:
        meta = json.load(f)
    codes, _, cshape, (yy, xx) = _categories(z, meta)
    cat = np.full(shape, -1, int)
    cat[yy[0]:yy[-1] + 1, xx[0]:xx[-1] + 1] = codes.reshape(cshape)
    pc = None
    if 'pixel_chi2' in z.files:
        pc = np.full(shape, np.nan)
        pc[yy[0]:yy[-1] + 1, xx[0]:xx[-1] + 1] = z['pixel_chi2']
    return cat, pc


def _stats(ci, ce):
    ok = np.isfinite(ci) & np.isfinite(ce) & (ci > 0)
    if ok.sum() == 0:
        return None
    r = ce[ok] / ci[ok]
    return dict(N=int(ok.sum()),
                chi2_inv_median=float(np.median(ci[ok])),
                chi2_emu_median=float(np.median(ce[ok])),
                ratio_median=float(np.median(r)),
                frac_emu_as_good=float(np.mean(r <= 1.01)),
                frac_emu_2x_worse=float(np.mean(r > 2.0)))


def compare(run_dir, inv_dir, emu_dir, out_dir, labels_file=None,
            ckpt_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    regs, obs, shape = _regions(run_dir)

    # per region, on the region's own grid; inactive pixels -> NaN
    per_region = {}
    for r in regs:
        o = obs[r['obs_file']]
        per_region[region_tag(r['obs_file'])] = dict(
            inv=_region_parts(run_dir, inv_dir, r['obs_file'], o),
            emu=_region_parts(run_dir, emu_dir, r['obs_file'], o),
            fine=int(o.lts[0]) < 0)
    # all regions combined per fine pixel (worst region, as for the labels)
    fi = labels._fine_pixel_chi2(run_dir, regs, *shape, out_dir=inv_dir)
    fe = labels._fine_pixel_chi2(run_dir, regs, *shape, out_dir=emu_dir)

    cat = None
    if labels_file and ckpt_dir:
        cat, pc = _category_map(labels_file, ckpt_dir, shape)
        if pc is not None:
            ok = np.isfinite(pc) & (pc > 0)
            dev = np.median(np.abs(fi[ok] - pc[ok]) / pc[ok])
            if dev > 1e-3:
                print(f'compare: WARNING the inversion synthetics in {inv_dir} '
                      f'do not reproduce the pixel chi2 stored in the label '
                      f'record (median relative difference {dev:.2g}): they '
                      'are probably not from the labelled atmosphere.')
            else:
                print('compare: inversion synthetics reproduce the label '
                      f'record\'s pixel chi2 (median rel. difference {dev:.1g})')

    summary = dict(run=run_dir, inv=inv_dir, emu=emu_dir, regions={},
                   all_regions={})
    head = (f'{"":22s} {"N":>6s} {"chi2 inv":>9s} {"chi2 emu":>9s} '
            f'{"emu/inv":>8s} {"as good":>9s} {"emu>2x":>7s}')

    def line(name, s):
        print(f'{name:22s} {s["N"]:6d} {s["chi2_inv_median"]:9.4g} '
              f'{s["chi2_emu_median"]:9.4g} {s["ratio_median"]:8.3f} '
              f'{100 * s["frac_emu_as_good"]:8.0f}% '
              f'{100 * s["frac_emu_2x_worse"]:6.0f}%')

    print('\nmedians over pixels; emu/inv = median of the per-pixel ratio; '
          'as good: fraction of pixels with emulator chi2 <= 1.01 x '
          'inversion chi2; emu>2x: more than twice the inversion chi2')
    print(head)
    rel_lines = []
    for tag, pr in per_region.items():
        s = _stats(pr['inv']['total'], pr['emu']['total'])
        if not s:
            continue
        summary['regions'][tag] = s
        line(tag, s)
        if pr['inv']['QUV'] is not None:
            for part in ('I', 'QUV'):
                sp = _stats(pr['inv'][part], pr['emu'][part])
                if sp:
                    s[f'stokes_{part}'] = sp
                    line(f'  Stokes {part}', sp)
        if pr['inv']['rel'] is not None:
            sr = _rel_stats(pr['inv']['rel'], pr['emu']['rel'])
            if sr:
                s['rel_intensity_error'] = sr
                rel_lines.append((tag, sr))
    s = _stats(fi, fe)
    summary['all_regions']['all'] = s
    line('all regions (worst)', s)
    if cat is not None:
        for i, c in enumerate(CATS):
            s = _stats(np.where(cat == i, fi, np.nan),
                       np.where(cat == i, fe, np.nan))
            if s:
                summary['all_regions'][c] = s
                line(f'  {c}', s)

    if rel_lines:
        print('\nsingle-wavelength regions: relative intensity error '
              '|obs - syn| / obs (median, 90th percentile)')
        for tag, sr in rel_lines:
            print(f'{tag:22s} inversion {100 * sr["rel_err_inv_median"]:.2f}% '
                  f'/ {100 * sr["rel_err_inv_p90"]:.2f}%   emulator '
                  f'{100 * sr["rel_err_emu_median"]:.2f}% / '
                  f'{100 * sr["rel_err_emu_p90"]:.2f}%   emulator closer in '
                  f'{100 * sr["frac_emu_closer"]:.0f}% of pixels')

    with open(os.path.join(out_dir, 'chi2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'\nwrote {os.path.join(out_dir, "chi2_summary.json")}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed: skipping figures')
        return summary
    _maps_figure(plt, per_region, obs, fi, fe, cat, out_dir)
    _scatter_figure(plt, fi, fe, cat, out_dir)
    _spectra_figure(plt, run_dir, inv_dir, emu_dir, regs, obs, shape,
                    fi, fe, cat, out_dir)
    return summary


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def _save(plt, fig, out_dir, name):
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f'wrote {p}')


def _maps_figure(plt, per_region, obs, fi, fe, cat, out_dir):
    """Per region: inversion chi2 | emulator chi2 | log10(emulator /
    inversion), and for full-Stokes regions the ratio of the Stokes I and
    the Q, U, V parts; single-wavelength regions show the relative
    intensity error instead. Last row: all regions combined (fine grid)."""
    from matplotlib.colors import LogNorm

    def lratio(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log10(b / a)

    rows = []
    for tag, pr in per_region.items():
        pi, pe = pr['inv'], pr['emu']
        if pi['rel'] is not None:
            rows.append((tag, 'rel', pi['rel'] * 100, pe['rel'] * 100, None,
                         None, pr['fine']))
        else:
            quv = pi['QUV'] is not None
            rows.append((tag, 'chi2', pi['total'], pe['total'],
                         lratio(pi['I'], pe['I']) if quv else None,
                         lratio(pi['QUV'], pe['QUV']) if quv else None,
                         pr['fine']))
    rows.append(('all regions (worst, fine grid)', 'chi2', fi, fe, None, None,
                 True))

    fig, axs = plt.subplots(len(rows), 5, figsize=(24, 4.0 * len(rows)),
                            constrained_layout=True)
    axs = np.atleast_2d(axs)
    for k, (tag, kind, a, b, lr_i, lr_q, fine) in enumerate(rows):
        if kind == 'chi2':
            both = np.r_[a[np.isfinite(a) & (a > 0)], b[np.isfinite(b) & (b > 0)]]
            lo, hi = np.percentile(both, [1, 99]) if both.size else (1, 10)
            panels = [(a, 'inversion chi2', 'magma', LogNorm(lo, hi)),
                      (b, 'emulator chi2', 'magma', LogNorm(lo, hi)),
                      (lratio(a, b), 'log10(emulator / inversion)', 'div', None),
                      (lr_i, 'Stokes I part: log10(emu / inv)', 'div', None),
                      (lr_q, 'Stokes Q,U,V part: log10(emu / inv)', 'div', None)]
        else:
            hi = float(np.nanpercentile(np.r_[a.ravel(), b.ravel()], 99))
            d = b - a
            panels = [(a, 'inversion |dI/I| [%]', 'magma', (0, hi)),
                      (b, 'emulator |dI/I| [%]', 'magma', (0, hi)),
                      (d, 'emulator - inversion [% points]', 'div%', None),
                      (None, '', '', None), (None, '', '', None)]
        for c, (img, title, cm, norm) in enumerate(panels):
            ax = axs[k, c]
            if img is None:
                ax.set_axis_off()
                continue
            if cm in ('div', 'div%'):
                lim = max(0.1, float(np.nanpercentile(np.abs(img), 99)))
                im = ax.imshow(img, origin='lower', cmap='RdBu_r',
                               vmin=-lim, vmax=lim)
            elif isinstance(norm, tuple):
                im = ax.imshow(img, origin='lower', cmap=cm, vmin=norm[0],
                               vmax=norm[1])
            else:
                im = ax.imshow(img, origin='lower', cmap=cm, norm=norm)
            fig.colorbar(im, ax=ax, shrink=0.85)
            ax.set_title(f'{tag}: {title}', fontsize=10)
            if fine and cat is not None and img.shape == cat.shape:
                ax.contour((cat == 3).astype(float), levels=[0.5],
                           colors='w' if c < 2 else 'k', linewidths=0.5,
                           alpha=0.7)
                ax.contour((cat == 1).astype(float), levels=[0.5],
                           colors='c' if c < 2 else 'k', linewidths=0.9,
                           linestyles='--')
    fig.suptitle('fit to the observations   (red: emulator fits worse, '
                 'blue: better;   solid outline: excluded by the chi2 filter, '
                 'dashed: held-out blocks)', fontsize=11)
    _save(plt, fig, out_dir, 'chi2_maps.png')


def _scatter_figure(plt, fi, fe, cat, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    a, b = fi.ravel(), fe.ravel()
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    cats = np.full(a.size, -1) if cat is None else cat.ravel()
    order = [(3, 'dropped'), (2, 'guard'), (0, 'train'), (1, 'held-out'),
             (-1, 'pixels' if cat is None else 'outside labels')]
    for i, name in order:
        sel = ok & (cats == i)
        if sel.any():
            ax.scatter(a[sel], b[sel], s=2, alpha=0.35, rasterized=True,
                       color=_CAT_COLORS[i] if i >= 0 else '0.4', label=name)
    lo, hi = np.percentile(np.r_[a[ok], b[ok]], [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], 'k-', lw=0.8)
    ax.plot([lo, hi], [2 * lo, 2 * hi], 'k:', lw=0.8, label='2x')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('inversion chi2 (worst region)')
    ax.set_ylabel('emulator chi2 (worst region)')
    ax.legend(markerscale=5, fontsize=8)
    ax.set_title('per fine pixel: below the diagonal the emulator fits better')
    _save(plt, fig, out_dir, 'chi2_scatter.png')


def _spectra_figure(plt, run_dir, inv_dir, emu_dir, regs, obs, shape,
                    fi, fe, cat, out_dir):
    """Observed vs synthetic profiles (inversion and emulator) at a few
    characteristic pixels; Stokes I, plus V where it is fitted."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = (fe / fi).ravel()
    ok = np.isfinite(ratio)
    cats = None if cat is None else cat.ravel()

    def pick(sel, how):
        idx = np.where(ok & sel)[0]
        if idx.size == 0:
            return None
        r = ratio[idx]
        j = {'max': np.argmax(r), 'min': np.argmin(r),
             'median': np.argmin(np.abs(r - np.median(r)))}[how]
        return int(idx[j])

    allpix = np.ones(ratio.size, bool)
    if cats is None:
        picks = [(pick(allpix, 'max'), 'emulator worst'),
                 (pick(allpix, 'median'), 'typical'),
                 (pick(allpix, 'min'), 'emulator best')]
    else:
        kept = (cats >= 0) & (cats <= 2)
        picks = [(pick(kept, 'max'), 'emulator worst (chi2-kept)'),
                 (pick(cats == 0, 'median'), 'typical training'),
                 (pick(cats == 1, 'median'), 'typical held-out'),
                 (pick(cats == 3, 'min'), 'emulator best (chi2-excluded)')]
    picks = [(i, w) for i, w in picks if i is not None]

    cols = []
    for r in regs:
        o = obs[r['obs_file']]
        used = (o.weights < 1e10).any(axis=0)
        for s, name in ((0, 'I'), (3, 'V')):
            if s < o.ns and used[s]:
                cols.append((r, s, name))
    syn = {r['obs_file']: (_read_out(run_dir, inv_dir, r['obs_file']),
                           _read_out(run_dir, emu_dir, r['obs_file']))
           for r in regs}
    fig, axs = plt.subplots(len(picks), len(cols),
                            figsize=(3.3 * len(cols), 2.6 * len(picks)),
                            constrained_layout=True, squeeze=False)
    for k, (i, what) in enumerate(picks):
        y, x = np.unravel_index(i, shape)
        for c, (r, s, name) in enumerate(cols):
            o = obs[r['obs_file']]
            if int(o.lts[0]) < 0:
                ry, rx = y, x
            else:
                row, col = labels.coarse_index(o, *shape)
                ry, rx = row[y, x], col[y, x]
            si, se = syn[r['obs_file']]
            ax = axs[k, c]
            w0 = float(np.round(np.median(o.wav)))
            w = o.wav - w0
            fit = o.weights[:, s] < 1e10          # wavelengths in the chi2
            sty = '-' if fit.sum() > 3 else 'o'   # continuum points
            ax.plot(w[fit], o.dat[0, ry, rx, fit, s], 'k.', ms=3,
                    label='observed')
            ax.plot(w[fit], si[ry, rx, fit, s], sty, color='#3b78c2',
                    lw=1.1, ms=5, mfc='none', label='inversion')
            ax.plot(w[fit], se[ry, rx, fit, s], sty, color='#c9473c',
                    lw=1.1, ms=5, mfc='none', label='emulator')
            if k == 0:
                ax.set_title(f'{region_tag(r["obs_file"])}  Stokes {name}'
                             f'\n(wavelength - {w0:.0f} A)', fontsize=9)
            if c == 0:
                ax.set_ylabel(f'{what}\n(y={y}, x={x})\nchi2 inv '
                              f'{fi[y, x]:.3g}, emu {fe[y, x]:.3g}',
                              fontsize=8)
            ax.tick_params(labelsize=7)
            if fit.sum() == 1:
                ax.set_xticks([])
            if k == len(picks) - 1:
                ax.set_xlabel('delta wavelength [A]', fontsize=8)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle('observed and synthetic profiles at characteristic pixels')
    _save(plt, fig, out_dir, 'chi2_spectra.png')


# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='step', required=True)
    p = sub.add_parser('model', help='emulator atmosphere for STiC mode 2')
    p.add_argument('--pred', required=True, help='predicted_atmos.nc')
    p.add_argument('--inv', required=True, help="the inversion's atmosphere")
    p.add_argument('--out', required=True)
    p = sub.add_parser('degrade', help='synthetic -> instrument grids')
    p.add_argument('--run', required=True, help='launch dir with input.cfg')
    p.add_argument('--synth', required=True, help='mode-2 profiles file')
    p.add_argument('--out', required=True, help='own directory for out_ files')
    p = sub.add_parser('compare', help='chi2 of emulator vs inversion')
    p.add_argument('--run', required=True, help='launch dir with input.cfg')
    p.add_argument('--inv', required=True, help='degraded inversion synthetics')
    p.add_argument('--emu', required=True, help='degraded emulator synthetics')
    p.add_argument('--out', required=True)
    p.add_argument('--labels', help='full_map.npz (training categories)')
    p.add_argument('--ckpt', help='checkpoint dir of the prediction')
    a = ap.parse_args()
    if a.step == 'model':
        synthesis_model(a.pred, a.inv, a.out)
    elif a.step == 'degrade':
        degrade(a.run, a.synth, a.out)
    else:
        compare(a.run, a.inv, a.emu, a.out, a.labels, a.ckpt)
