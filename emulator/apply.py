"""
Apply a trained emulator to new observations — no inversion needed.

Each input is a run dir prepared exactly like for a coupled STiC inversion
of those data (obs_*.nc with psf/lts/ltargs/ds, inst_*.nc, input.cfg with
the region lines); one run dir per frame. The ensemble is loaded once.

  python -m emulator.apply --ckpt <ckpts> --out <dir> <run_dir> [<run_dir> ...]
      [--pgas-top 1.0 | --boundary-model <starting model .nc>]

Per run dir, <out>/<run dir name>/ receives
  input_check.json  how the new data compare with the training inputs
  predicted_atmos.nc STiC model on the inversion depth grid; the top gas
                    pressure (hydrostatic boundary) is --pgas-top or taken
                    from --boundary-model. Use it for a mode-2 synthesis
                    (chi2_compare) or as input_model of a coupled inversion.
  prediction.npz    ensemble spread std_<var> on the same grid, ...
  nyquist.json      checkerboard diagnostic
  quicklook.png     maps at log tau -1/-3/-5 with the ensemble spread

Input check: the network only knows the spectral regions, wavelengths and
Stokes parameters it was trained on, in the same intensity calibration.
A different region layout or wavelength set stops the run; intensity
levels far outside the training range are reported (the prediction is
then an extrapolation, which the ensemble spread may underestimate).
"""

import os
import json
import argparse
import numpy as np

from . import tiles, stic_io
from . import nn_dataset as nd
from .labels import region_tag

# |z| (training standard deviations) above which an input counts as outside
# the training range, and the fractions / shifts that trigger a warning
Z_OUT = 5.0
WARN_FRAC = 0.05
WARN_SHIFT = 1.0


def check_inputs(run_dir, enc):
    """Compare a run dir's inputs with the training inputs. Raises on a
    layout the network cannot take; returns a per-region report."""
    with open(os.path.join(run_dir, 'input.cfg')) as f:
        regs = tiles.parse_regions(f.read())
    have = {region_tag(r['obs_file']): r for r in regs}
    trained = sorted(k[8:] for k in enc.files if k.startswith('stat_mu_'))
    missing = [t for t in trained if t not in have]
    if missing:
        raise ValueError(f'{run_dir}: regions {missing} are missing; the '
                         f'network was trained on {trained} (region tag = '
                         'obs file name without extension)')
    extra = [t for t in have if t not in trained]
    if extra:
        print(f'check: regions {extra} are not used by the network')
    pol = bool(enc['pol_over_i']) if 'pol_over_i' in enc.files else False

    report = {}
    for tag in trained:
        o = tiles.CoupledObs(os.path.join(run_dir, have[tag]['obs_file']))
        w, s = nd._used_channels(o.weights)
        n_tr = enc[f'stat_mu_{tag}'].size
        if w.size != n_tr:
            raise ValueError(
                f'{tag}: {w.size} input channels (wavelength x Stokes with '
                f'weight < 1e10), the network expects {n_tr}. Prepare the '
                'obs file with the training wavelengths and Stokes weights.')
        rep = {}
        if f'chan_wav_{tag}' in enc.files:
            wt = np.asarray(enc[f'chan_wav_{tag}'], 'float64')
            st = np.asarray(enc[f'chan_stokes_{tag}'])
            if not np.array_equal(st, s):
                raise ValueError(f'{tag}: the fitted Stokes parameters differ '
                                 'from the training data')
            dw = float(np.abs(o.wav[w] - wt).max())
            sp = np.diff(np.unique(wt))
            tol = 0.1 * sp.min() if sp.size else 1e-3
            rep['max_wavelength_offset'] = dw
            if dw > tol:
                raise ValueError(
                    f'{tag}: wavelengths differ from the training data by up '
                    f'to {dw:.4g} (tolerance {tol:.2g}); resample the '
                    'observations onto the training wavelengths')
        if o.nt > 1:
            print(f'check: {tag} holds {o.nt} time steps; only the first is '
                  'used (prepare one run dir per frame)')
        # intensity levels in units of the training spread
        img = nd.region_channels(o.dat[0], o.weights, pol)
        mu, sd = enc[f'stat_mu_{tag}'], enc[f'stat_sd_{tag}']
        z = (img - mu[:, None, None]) / sd[:, None, None]
        act = (o.pweights[0] > 0) & np.isfinite(z).all(axis=0)
        zi = z[s == 0][:, act]
        rep['stokes_i_shift'] = float(np.median(zi)) if zi.size else 0.0
        rep['frac_outside'] = float(np.mean(
            (np.abs(z[:, act]) > Z_OUT).any(axis=0))) if act.any() else 0.0
        report[tag] = rep
        flag = ''
        if abs(rep['stokes_i_shift']) > WARN_SHIFT or \
                rep['frac_outside'] > WARN_FRAC:
            flag = ('   <- WARNING: outside the training range (other '
                    'calibration, or a scene unlike the training data)')
        print(f'check: {tag:12s} Stokes I level {rep["stokes_i_shift"]:+.2f}'
              f' training sd, {100 * rep["frac_outside"]:.1f}% of pixels '
              f'with an input beyond {Z_OUT:.0f} sd{flag}')
    return report


def boundary_from_model(path, shape):
    """Top gas pressure per pixel from a STiC model (e.g. the starting
    model); a single-pixel or uniform model is broadcast."""
    m = stic_io.Model.read(path, t0=0, t1=1)
    top = m.pgas[0][..., 0]
    if top.shape == shape:
        return top
    if np.ptp(top) == 0:
        return float(top.flat[0])
    raise ValueError(f'{path}: {top.shape} pixels with varying top gas '
                     f'pressure cannot be mapped onto the {shape} FOV')


def _quicklook(run_out, ltaus=(-1.0, -3.0, -5.0)):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    m = stic_io.Model.read(os.path.join(run_out, 'predicted_atmos.nc'))
    pz = np.load(os.path.join(run_out, 'prediction.npz'))
    lt = m.ltau[0, 0, 0].astype('float64')
    rows = [('temp', 'temp', 'T [K]', 1.0, 'inferno', False),
            ('vlos', 'vlos', 'v_los [km/s]', 1e-5, 'RdBu_r', True),
            ('blong', 'Bln', 'B_long [G]', 1.0, 'RdGy_r', True),
            ('bhor', 'Bho', 'B_hor [G]', 1.0, 'viridis', False)]
    fig, axs = plt.subplots(len(rows), 2 * len(ltaus),
                            figsize=(4.0 * len(ltaus) * 2, 3.4 * len(rows)),
                            constrained_layout=True, sharex=True, sharey=True)
    for r, (v, attr, lab, sc, cmap, sym) in enumerate(rows):
        cube = getattr(m, attr)[0].astype('float64') * sc
        std = pz[f'std_{v}'].astype('float64') * sc if f'std_{v}' in pz.files \
            else None
        for c, t in enumerate(ltaus):
            A = nd.interp_matrix([t], lt)[0]
            img = cube @ A
            ax = axs[r, 2 * c]
            if sym:
                hi = np.nanpercentile(np.abs(img), 99)
                im = ax.imshow(img, origin='lower', cmap=cmap, vmin=-hi, vmax=hi)
            else:
                lo, hi = np.nanpercentile(img, [1, 99])
                im = ax.imshow(img, origin='lower', cmap=cmap, vmin=lo, vmax=hi)
            fig.colorbar(im, ax=ax, shrink=0.85)
            ax.set_title(f'{lab}  log tau {t:+.0f}', fontsize=9)
            ax = axs[r, 2 * c + 1]
            if std is not None:
                im = ax.imshow(std @ A, origin='lower', cmap='magma')
                fig.colorbar(im, ax=ax, shrink=0.85)
                ax.set_title('ensemble spread', fontsize=9)
            else:
                ax.set_axis_off()
    fig.suptitle(f'emulator prediction: {os.path.basename(run_out)}')
    p = os.path.join(run_out, 'quicklook.png')
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f'wrote {p}')


def apply(run_dirs, ckpt_dir, out_root, pgas_top=1.0, boundary_model=None):
    from .train_net import load_ensemble, _device
    from .predict_map import predict
    device = _device()
    models, enc = load_ensemble(ckpt_dir, device)
    outs = []
    for run_dir in run_dirs:
        name = os.path.basename(os.path.normpath(run_dir))
        run_out = os.path.join(out_root, name)
        os.makedirs(run_out, exist_ok=True)
        print(f'\n=== {run_dir} -> {run_out}')
        report = check_inputs(run_dir, enc)
        top = pgas_top
        if boundary_model:
            _, shape = nd.rundir_to_sample(run_dir, {
                k[8:]: (enc[f'stat_mu_{k[8:]}'], enc[f'stat_sd_{k[8:]}'])
                for k in enc.files if k.startswith('stat_mu_')})
            top = boundary_from_model(boundary_model, shape)
        predict(run_dir, ckpt_dir, run_out, pgas_top=top,
                ensemble=(models, enc, device))
        with open(os.path.join(run_out, 'input_check.json'), 'w') as f:
            json.dump(dict(run_dir=run_dir, ckpt=ckpt_dir, regions=report,
                           pgas_top=boundary_model or float(pgas_top)),
                      f, indent=1)
        _quicklook(run_out)
        outs.append(run_out)
    return outs


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dirs', nargs='+')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--pgas-top', type=float, default=1.0,
                   help='top gas pressure for all pixels (default 1.0)')
    g.add_argument('--boundary-model',
                   help='STiC model whose top gas pressure is used '
                        '(e.g. the starting model of an inversion)')
    a = ap.parse_args()
    apply(a.run_dirs, a.ckpt, a.out, a.pgas_top, a.boundary_model)
