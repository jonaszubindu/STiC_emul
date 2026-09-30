"""
Verify tile extraction: the tile's degradation operators must reproduce
the windowed full-FOV operators exactly for observed pixels whose whole
footprint (PSF wings + warp + rebin cell) lies inside the tile.

Also checks the coverage-based pixel_weights masking and the fine-grid
(identity) regions.
"""

import os
import shutil
import numpy as np

from emulator import degradation as dg
from emulator import tiles

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get('STIC_EMUL_TEST_DATA', '')
OUT = os.path.join(HERE, '.test_scratch', 'tile_test')

if not (SRC and os.path.isfile(os.path.join(SRC, 'input.cfg'))):
    import sys
    print('SKIP: set STIC_EMUL_TEST_DATA to a coupled run dir with input.cfg')
    sys.exit(0)

# geometry and region list from the run dir itself
from netCDF4 import Dataset
with open(os.path.join(SRC, 'input.cfg')) as f:
    _cfg = f.read()
REGIONS = [r['obs_file'] for r in tiles.parse_regions(_cfg)]
with Dataset(os.path.join(SRC, tiles.read_cfg_key(_cfg, 'input_model'))) as f:
    MODEL_NY = len(f.dimensions['y'])
    MODEL_NX = len(f.dimensions['x'])
TNX = min(48, MODEL_NX // 2)
TNY = min(44, MODEL_NY // 2)
MX0 = (MODEL_NX - TNX) // 3
MY0 = (MODEL_NY - TNY) // 3
print(f'model {MODEL_NY}x{MODEL_NX}, tile {TNY}x{TNX} at ({MY0},{MX0}), '
      f'regions {REGIONS}')

shutil.rmtree(OUT, ignore_errors=True)
rng = np.random.default_rng(11)

os.makedirs(os.path.join(HERE, '.test_scratch'), exist_ok=True)
meta = tiles.extract_tile_rundir(SRC, OUT, MX0, MY0, TNX, TNY,
                                 require_aux=False)
print('tile meta:', {k: v for k, v in meta['regions'].items()})

img_full = rng.normal(1.0, 0.3, (MODEL_NY, MODEL_NX))
# make it smooth-ish so numbers are O(1)
from scipy.ndimage import gaussian_filter
img_full = gaussian_filter(img_full, 2.0)
img_tile = img_full[MY0:MY0 + TNY, MX0:MX0 + TNX]

for reg in REGIONS:
    base = os.path.basename(reg)      # tile run dirs are flat
    D_full = dg.build_region_operator(os.path.join(SRC, reg),
                                      MODEL_NX, MODEL_NY)
    D_tile = dg.build_region_operator(os.path.join(OUT, base), TNX, TNY)

    ox0, ox1, oy0, oy1 = meta['regions'][base]['window']
    ny_t, nx_t = meta['regions'][base]['ny'], meta['regions'][base]['nx']

    y_full = (D_full @ img_full.ravel()).reshape(-1)
    with np.errstate(all='ignore'):
        y_tile = (D_tile @ img_tile.ravel()).reshape(ny_t, nx_t)

    # full-op output restricted to the tile's observed window
    obs = tiles.CoupledObs(os.path.join(SRC, reg))
    y_full = y_full.reshape(obs.ny, obs.nx)[oy0:oy1, ox0:ox1]

    # The load-bearing property: every obs pixel that PARTICIPATES in the
    # tile inversion (pixel_weight > 0) must have an operator row
    # identical to the full-FOV one.
    # tile_obs masks every pixel whose operator row cannot be exact
    # (coarse: dual coverage; fine with PSF: PSF coverage), so
    # pweights > 0 is the exactness guarantee for all region types
    tob = tiles.CoupledObs(os.path.join(OUT, base))
    active = tob.pweights[0] > 0

    err = np.abs(y_tile - y_full)[active]
    print(f'{reg}: obs window x[{ox0}:{ox1}] y[{oy0}:{oy1}] '
          f'tile obs {ny_t}x{nx_t}, active px {active.sum()}/{active.size}, '
          f'max err on active = {err.max():.3e}')
    assert active.sum() > 0.4 * active.size, f'{reg}: too few active pixels'
    assert err.max() < 1e-12, reg

# --------------------------------------------------------------------------- #
# per-pixel instrumental profiles: STiC indexes psf(yy, xx, :) with the
# run's own pixel coordinates, so the tile must carry the SLICED profiles
# --------------------------------------------------------------------------- #
n_checked = 0
for r in tiles.parse_regions(_cfg):
    src_inst = os.path.join(SRC, r['inst_file'])
    if not os.path.isfile(src_inst):
        continue
    dst_inst = os.path.join(OUT, os.path.basename(r['inst_file']))
    assert os.path.isfile(dst_inst) and not os.path.islink(dst_inst), dst_inst
    with Dataset(src_inst) as f:
        full = np.ma.filled(f.variables['iprof'][:], np.nan)
    with Dataset(dst_inst) as f:
        tile = np.ma.filled(f.variables['iprof'][:], np.nan)
    if full.ndim >= 2:
        want = full[MY0:MY0 + TNY, MX0:MX0 + TNX]
        assert tile.shape == want.shape, (dst_inst, tile.shape, want.shape)
        assert np.array_equal(tile, want, equal_nan=True), dst_inst
        # a symlinked/unsliced file would have served (0:TNY, 0:TNX)
        wrong = full[:TNY, :TNX]
        print(f'{os.path.basename(dst_inst)}: per-pixel iprof sliced '
              f'{full.shape} -> {tile.shape}; max |tile-local mix-up| '
              f'would have been {np.nanmax(np.abs(wrong - want)):.3e}')
    else:
        assert np.array_equal(tile, full, equal_nan=True), dst_inst
        print(f'{os.path.basename(dst_inst)}: 1-D iprof copied')
    n_checked += 1
print(f'instrumental profiles checked: {n_checked}')

# --------------------------------------------------------------------------- #
# inversion_mask on the model grid is sliced too (temporary launch dir)
# --------------------------------------------------------------------------- #
LD = os.path.join(HERE, '.test_scratch', 'mask_launch')
shutil.rmtree(LD, ignore_errors=True)
os.makedirs(LD)
for e in os.listdir(SRC):
    if e != 'input.cfg':
        os.symlink(os.path.abspath(os.path.join(SRC, e)), os.path.join(LD, e))
mask_full = (np.arange(MODEL_NY * MODEL_NX) % 7).reshape(MODEL_NY, MODEL_NX)
with Dataset(os.path.join(LD, 'mask_test.nc'), 'w') as f:
    f.createDimension('nx', MODEL_NX)
    f.createDimension('ny', MODEL_NY)
    f.createVariable('mask', 'i4', ('ny', 'nx'))[:] = mask_full
with open(os.path.join(LD, 'input.cfg'), 'w') as f:
    f.write(_cfg + '\ninversion_mask = mask_test.nc\n')
OUTM = os.path.join(HERE, '.test_scratch', 'tile_mask')
shutil.rmtree(OUTM, ignore_errors=True)
tiles.extract_tile_rundir(LD, OUTM, MX0, MY0, TNX, TNY, require_aux=False)
with Dataset(os.path.join(OUTM, 'mask_test.nc')) as f:
    mt = f.variables['mask'][:]
assert np.array_equal(mt, mask_full[MY0:MY0 + TNY, MX0:MX0 + TNX])
with open(os.path.join(OUTM, 'input.cfg')) as f:
    assert tiles.read_cfg_key(f.read(), 'inversion_mask') == 'mask_test.nc'
print('inversion_mask: sliced and cfg rewritten')

print('\nALL TILE TESTS PASSED')
