"""
Apply a trained coupled-emulator ensemble to a full run dir and write:
  predicted_atmos.nc   depth-stratified STiC model on the fine grid
  prediction.npz       standardized means + aleatoric/epistemic std maps
                       + the Nyquist power-spectrum diagnostic

The power-spectrum diagnostic addresses the checkerboard null space:
chi2 through the degradation operators is blind to fine-grid structure
the instruments cannot see, so we compare the azimuthally averaged
spatial power of each predicted parameter map near the coarse grids'
Nyquist frequencies against the low-frequency power. Ratios far above
the label tiles' typical values flag null-space junk.

Usage:
  python predict_map.py <run_dir> --ckpt <ckpt_dir> --out <out_dir>
"""

import os
import sys
import argparse
import numpy as np
import torch

from . import emu_config as C
from . import nn_dataset as nd
from . import tiles
from .train_net import load_ensemble, _device

from . import stic_io

PREDICT_NDEP = 55


def power_spectrum_1d(img):
    """Azimuthally averaged power spectrum of a 2-D map. Returns (k, P)
    with k in cycles/pixel."""
    f = np.fft.fftshift(np.abs(np.fft.fft2(img - np.nanmean(img))) ** 2)
    ny, nx = img.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    kr = np.hypot(ky[:, None], kx[None, :])
    bins = np.linspace(0, 0.5, 26)
    P = np.zeros(bins.size - 1)
    for i in range(bins.size - 1):
        m = (kr >= bins[i]) & (kr < bins[i + 1])
        P[i] = f[m].mean() if m.any() else 0.0
    return 0.5 * (bins[:-1] + bins[1:]), P


def nyquist_diagnostic(maps, coarse_cells):
    """Ratio of power near each coarse grid's Nyquist (k = 1/(2*cell))
    to low-frequency power, per parameter map."""
    out = {}
    for name, img in maps.items():
        k, P = power_spectrum_1d(np.nan_to_num(img))
        low = P[(k > 0.02) & (k < 0.1)].mean()
        ratios = {}
        for tag, cell in coarse_cells.items():
            knyq = 0.5 / cell
            sel = (k > 0.8 * knyq) & (k < min(1.2 * knyq, 0.5))
            ratios[tag] = float(P[sel].mean() / max(low, 1e-30)) \
                if sel.any() else np.nan
        out[name] = ratios
    return out


def _encoding_layout(enc):
    """(names, positions per channel, output depth grid, pol_over_i,
    linear continuation per channel) for any checkpoint; older ones
    (fixed 16-point grid, no stored layout) fall back to their LTAU_GRID
    and the former 55-point output grid, constant beyond the ends."""
    names = [str(s) for s in enc['target_names']]
    grid = np.asarray(enc['ltau_grid'], 'float64')
    if 'target_ltau' in enc.files:
        where = np.asarray(enc['target_ltau'], 'float64')
    else:
        where = np.tile(grid, len(names) // grid.size)
    if 'depth_grid' in enc.files:
        depth = np.asarray(enc['depth_grid'], 'float64')
    else:
        depth = np.linspace(grid[0], grid[-1], PREDICT_NDEP)
    pol = bool(enc['pol_over_i']) if 'pol_over_i' in enc.files else False
    extrap = np.asarray(enc['target_extrap'], bool) \
        if 'target_extrap' in enc.files else np.zeros(len(names), bool)
    return names, where, depth, pol, extrap


def _member_spread(phys_members):
    """Ensemble spread per physical quantity (azimuth: circular spread of
    the doubled angle, i.e. respecting the 180-degree ambiguity)."""
    out = {}
    for v in phys_members[0]:
        st = np.stack([p[v] for p in phys_members])
        if v == 'azi':
            r = np.abs(np.exp(2j * st).mean(0))
            out[v] = np.sqrt(-2.0 * np.log(np.clip(r, 1e-12, 1.0))) / 2.0
        else:
            out[v] = st.std(0)
    return out


def predict(run_dir, ckpt_dir, out_dir, pgas_top=1.0, ensemble=None):
    """pgas_top: gas pressure at the top of the atmosphere (scalar or
    (ny, nx)), STiC's boundary for hydrostatic equilibrium; use the value
    of the starting model an inversion of these data would get.
    ensemble: (models, enc, device) already loaded (several run dirs)."""
    os.makedirs(out_dir, exist_ok=True)
    if ensemble is None:
        device = _device()
        models, enc = load_ensemble(ckpt_dir, device)
    else:
        models, enc, device = ensemble
    stats = {k[8:]: (enc[f'stat_mu_{k[8:]}'], enc[f'stat_sd_{k[8:]}'])
             for k in enc.files if k.startswith('stat_mu_')}
    names, where, depth, pol, extrap = _encoding_layout(enc)
    ym, ys = enc['y_mean'], enc['y_std']

    sample, (ny, nx) = nd.rundir_to_sample(run_dir, stats, pol_over_i=pol)
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    coords = np.stack([yy.ravel(), xx.ravel()], 1).astype('float64')

    n_out = len(names)
    mus = np.empty((len(models), ny * nx, n_out), 'float32')
    avr = np.empty_like(mus)
    with torch.no_grad():
        for i0 in range(0, len(coords), C.PREDICT_CHUNK):
            cc = coords[i0:i0 + C.PREDICT_CHUNK]
            for k, m in enumerate(models):
                mu, lv = m(sample, cc, device)
                mus[k, i0:i0 + len(cc)] = mu.cpu().numpy()
                avr[k, i0:i0 + len(cc)] = np.exp(lv.cpu().numpy())
            print(f'predict: {min(i0 + C.PREDICT_CHUNK, len(coords))}'
                  f'/{len(coords)} px')
    mean = mus.mean(0) * ys + ym
    epi = np.sqrt(mus.var(0)) * ys
    ale = np.sqrt(avr.mean(0)) * ys

    # stratifications on the output depth grid, built like STiC builds
    # them from nodes (the inversion's own grid for node-based checkpoints)
    phys = nd.stratify(mean, names, where, depth, extrap)
    spread = _member_spread([nd.stratify(mus[k] * ys + ym, names, where,
                                         depth, extrap)
                             for k in range(len(models))])

    m = stic_io.Model(nt=1, ny=ny, nx=nx, ndep=depth.size)
    m.ltau[:] = depth[None, None, None, :]
    # STiC recomputes the pressure stratification by hydrostatic
    # equilibrium from T and the top gas pressure: a constant column
    # carrying the boundary value is all it needs
    m.pgas[0] = np.broadcast_to(np.asarray(pgas_top, 'float32')[..., None],
                                (ny, nx, depth.size)) \
        if np.ndim(pgas_top) else float(pgas_top)
    for v, arr in phys.items():
        a = stic_io._ATTR.get(v, v)
        getattr(m, a)[0] = arr.reshape(ny, nx, -1)
    m.vturb[0] = np.maximum(m.vturb[0], 0.0)
    out_nc = os.path.join(out_dir, 'predicted_atmos.nc')
    m.write(out_nc, write_all=False)

    # Nyquist diagnostic: coarse cell sizes from the region geometry
    coarse_cells = {}
    for tag, reg in sample['regions'].items():
        if reg['rd'][0] > 1.001:
            coarse_cells[tag] = float(reg['rd'][0])
    idep = int(np.argmin(np.abs(depth - (-1.0))))
    diag_maps = {
        'temp@ltau-1': phys['temp'][:, idep].reshape(ny, nx),
        'vlos@ltau-1': phys['vlos'][:, idep].reshape(ny, nx),
        'blong@ltau-1': phys['blong'][:, idep].reshape(ny, nx)}
    nyq = nyquist_diagnostic(diag_maps, coarse_cells)
    for name, r in nyq.items():
        print(f'nyquist {name}: ' + '  '.join(f'{t}={v:.3f}'
                                              for t, v in r.items()))

    np.savez_compressed(
        os.path.join(out_dir, 'prediction.npz'),
        mean_std_units=mus.mean(0), epistemic_std=epi.reshape(ny, nx, -1),
        aleatoric_std=ale.reshape(ny, nx, -1),
        target_names=np.asarray(names), target_ltau=where,
        depth_grid=depth, ltau_grid=np.asarray(enc['ltau_grid']),
        **{f'std_{v}': a.reshape(ny, nx, -1).astype('float32')
           for v, a in spread.items()})
    with open(os.path.join(out_dir, 'nyquist.json'), 'w') as f:
        import json
        json.dump(nyq, f, indent=1)
    print(f'predict: wrote {out_nc}')
    return out_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pgas-top', type=float, default=1.0,
                    help='top gas pressure (hydrostatic boundary)')
    a = ap.parse_args()
    predict(a.run_dir, a.ckpt, a.out, pgas_top=a.pgas_top)
