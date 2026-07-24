"""
Verify the Python/torch degradation operators against the C++ ground
truth (gt_dump, which includes STiC's own linear_transformations.hpp).

Run inside the nn_emulator venv:
  python test_degradation.py
"""

import os
import sys
import subprocess
import numpy as np
import scipy.sparse as sp

from emulator import degradation as dg

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, 'gt_dump')
RUN_DIR = os.environ.get('STIC_EMUL_TEST_DATA', '')
STIC_SRC = os.environ.get('STIC_SRC', '')
SCRATCH = os.path.join(HERE, '.test_scratch')

if not (RUN_DIR and os.path.isdir(RUN_DIR)):
    print('SKIP: set STIC_EMUL_TEST_DATA to a coupled run dir '
          '(obs_*.nc with lts/ltargs/ds)')
    sys.exit(0)
if not os.path.isfile(GT):
    if not (STIC_SRC and os.path.isfile(os.path.join(
            STIC_SRC, 'linear_transformations.hpp'))):
        print('SKIP: set STIC_SRC to coupled_stic/src to build gt_dump')
        sys.exit(0)
    subprocess.run(['clang++', '-O2', '-std=c++14', f'-I{STIC_SRC}',
                    '-I/usr/local/include',
                    os.path.join(HERE, 'gt_dump.cc'), '-o', GT], check=True)
os.makedirs(SCRATCH, exist_ok=True)
rng = np.random.default_rng(3)


def gt_matrix(args, shape):
    """Run gt_dump and parse the triplets into a CSR matrix."""
    out = subprocess.run([GT] + [str(a) for a in args],
                         capture_output=True, text=True, check=True)
    rows, cols, vals = [], [], []
    for ln in out.stdout.splitlines():
        r, c, v = ln.split()
        rows.append(int(r)); cols.append(int(c)); vals.append(float(v))
    return sp.csr_matrix((vals, (rows, cols)), shape=shape)


def compare(name, mine, gt, tol=1e-12):
    d = (mine - gt).tocoo()
    err = np.abs(d.data).max() if d.nnz else 0.0
    scale = np.abs(gt.data).max()
    rel = err / scale if scale > 0 else err
    status = 'OK' if rel < tol else 'FAIL'
    print(f'{name:28s} nnz(mine)={mine.nnz:8d} nnz(gt)={gt.nnz:8d} '
          f'max rel err={rel:.3e}  {status}')
    assert rel < tol, name
    return rel


# --------------------------------------------------------------------------- #
# 1. PSF (asymmetric, to catch orientation/flip errors)
# --------------------------------------------------------------------------- #
nx, ny = 17, 13
npx, npy = 7, 5
psf = rng.uniform(0.01, 1.0, (npy, npx))
psf[0, 0] = 5.0            # strong asymmetry
pf = os.path.join(SCRATCH, 'psf.bin')
psf.astype('float64').tofile(pf)

gt = gt_matrix(['psf', nx, ny, npx, npy, pf], (nx * ny, nx * ny))
mine = dg.psf_matrix(nx, ny, psf)
compare('psf (asymmetric)', mine, gt)

# even-sized PSF (center convention npx//2)
psf2 = rng.uniform(0.01, 1.0, (4, 6))
pf2 = os.path.join(SCRATCH, 'psf2.bin')
psf2.astype('float64').tofile(pf2)
gt = gt_matrix(['psf', nx, ny, 6, 4, pf2], (nx * ny, nx * ny))
compare('psf (even-sized)', dg.psf_matrix(nx, ny, psf2), gt)

# --------------------------------------------------------------------------- #
# 2. rebin — non-integer scale factor and offsets
# --------------------------------------------------------------------------- #
rx, ry = 11, 9
nx2, ny2 = 29, 31
rdx, rdy = 2.63, 3.41
rsx, rsy = 0.7, -0.3
dx, dy, sx, sy = 1.0, 1.0, 0.0, 0.0
gt = gt_matrix(['rebin', rx, ry, rdx, rdy, rsx, rsy,
                nx2, ny2, dx, dy, sx, sy], (rx * ry, nx2 * ny2))
mine = dg.rebin_matrix(rx, ry, rdx, rdy, rsx, rsy, nx2, ny2, dx, dy, sx, sy)
compare('rebin (non-integer)', mine, gt)

# --------------------------------------------------------------------------- #
# 3. destretch — smooth random displacement field incl. boundary effects
# --------------------------------------------------------------------------- #
nx3, ny3 = 21, 25
yy, xx = np.mgrid[0:ny3, 0:nx3]
dsx = 1.5 * np.sin(2 * np.pi * yy / ny3) + rng.normal(0, 0.2, (ny3, nx3))
dsy = -1.2 * np.cos(2 * np.pi * xx / nx3) + rng.normal(0, 0.2, (ny3, nx3))
dsf = os.path.join(SCRATCH, 'ds.bin')
np.concatenate([dsx.ravel(), dsy.ravel()]).astype('float64').tofile(dsf)
gt = gt_matrix(['destretch', nx3, ny3, dsf], (nx3 * ny3, nx3 * ny3))
mine = dg.destretch_matrix(nx3, ny3, dsx, dsy)
compare('destretch (random field)', mine, gt)

# integer shifts (degenerate bilinear weights)
dsx_i = np.full((ny3, nx3), 2.0)
dsy_i = np.full((ny3, nx3), -3.0)
dsf2 = os.path.join(SCRATCH, 'ds2.bin')
np.concatenate([dsx_i.ravel(), dsy_i.ravel()]).astype('float64').tofile(dsf2)
gt = gt_matrix(['destretch', nx3, ny3, dsf2], (nx3 * ny3, nx3 * ny3))
compare('destretch (integer shift)', dg.destretch_matrix(nx3, ny3, dsx_i, dsy_i), gt)

# --------------------------------------------------------------------------- #
# 4. bintrim (LT type 4)
# --------------------------------------------------------------------------- #
nix, niy = 40, 35
btdx, btdy, btcx, btcy = 2.71, 1.93, 3.2, 1.1
nxe, nye = 12.0, 14.0
nox, noy = int(round(nxe)) - 1, int(round(nye)) - 1
gt = gt_matrix(['bintrim', nix, niy, btdx, btdy, btcx, btcy, nxe, nye],
               (nox * noy, nix * niy))
mine = dg.bintrim_matrix(nix, niy, btdx, btdy, btcx, btcy, nxe, nye)
compare('bintrim (type 4)', mine, gt)

# --------------------------------------------------------------------------- #
# 5. full composite operators from the real run directory
# --------------------------------------------------------------------------- #
from netCDF4 import Dataset

MODEL_NX = MODEL_NY = 140

for reg in ('obs_8542.nc', 'obs_6302.nc', 'obs_3934.nc'):
    fobs = os.path.join(RUN_DIR, reg)
    with Dataset(fobs) as f:
        ny_o = len(f.dimensions['y']); nx_o = len(f.dimensions['x'])
        has_psf = 'psf' in f.variables and 0 not in f.variables['psf'].shape
        psf_v = np.ma.filled(f.variables['psf'][:], 0.) if has_psf else None
        lts_v = np.ravel(np.ma.filled(f.variables['lts'][:], -1)) \
            if 'lts' in f.variables else np.array([-1])
        ltargs_v = np.ravel(np.ma.filled(f.variables['ltargs'][:], 0.)) \
            if 'ltargs' in f.variables else np.zeros(1)
        ds_v = np.ma.filled(f.variables['ds'][:], 0.) \
            if 'ds' in f.variables and 0 not in f.variables['ds'].shape else None

    # ---- ground truth composite, assembled from gt_dump pieces ------------ #
    if psf_v is not None and psf_v.sum() >= 1e-12:
        pfr = os.path.join(SCRATCH, 'psf_r.bin')
        np.asarray(psf_v, 'float64').tofile(pfr)
        gt_D = gt_matrix(['psf', MODEL_NX, MODEL_NY,
                          psf_v.shape[1], psf_v.shape[0], pfr],
                         (MODEL_NX * MODEL_NY, MODEL_NX * MODEL_NY))
    else:
        gt_D = sp.identity(MODEL_NX * MODEL_NY, format='csr')

    if int(lts_v[0]) >= 0:
        off = 0
        for ltid in lts_v.astype(int):
            a = ltargs_v[off:off + dg.LT_NARGS[ltid]]
            off += dg.LT_NARGS[ltid]
            if ltid == 10:
                nxi, nyi = int(a[6]), int(a[7])
                dsr = np.asarray(ds_v, 'float64').reshape(2, nyi, nxi)
                dsfr = os.path.join(SCRATCH, 'ds_r.bin')
                np.concatenate([dsr[0].ravel(), dsr[1].ravel()]).tofile(dsfr)
                gt_W = gt_matrix(['destretch', nxi, nyi, dsfr],
                                 (nxi * nyi, nxi * nyi))
                gt_R = gt_matrix(['rebin', int(a[0]), int(a[1]), a[2], a[3],
                                  a[4], a[5], nxi, nyi, a[8], a[9], a[10],
                                  a[11]], (int(a[0]) * int(a[1]), nxi * nyi))
                gt_D = (gt_R @ gt_W @ gt_D).tocsr()
            else:
                raise RuntimeError(f'unexpected LT {ltid} in {reg}')

    # ---- my implementation ------------------------------------------------ #
    mine_D = dg.build_region_operator(fobs, MODEL_NX, MODEL_NY)
    assert mine_D.shape == (ny_o * nx_o, MODEL_NX * MODEL_NY), \
        (reg, mine_D.shape, (ny_o * nx_o, MODEL_NX * MODEL_NY))
    compare(f'composite {reg}', mine_D, gt_D, tol=1e-11)

    # ---- application check: numpy vs torch, random image ------------------ #
    import torch
    op = dg.TorchRegionOperator(mine_D, (ny_o, nx_o), (MODEL_NY, MODEL_NX),
                                dtype=torch.float64)
    img = rng.normal(1.0, 0.3, (3, MODEL_NY, MODEL_NX))   # 3 "wavelengths"
    out_np = (mine_D @ img.reshape(3, -1).T).T.reshape(3, ny_o, nx_o)
    out_th = op.apply(torch.from_numpy(img)).numpy()
    aerr = np.abs(out_np - out_th).max()
    print(f'  apply: torch vs scipy max abs err = {aerr:.3e}')
    assert aerr < 1e-12

print('\nALL DEGRADATION TESTS PASSED')
