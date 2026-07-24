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
OUT = os.path.join(HERE, '.test_scratch', 'tile_36_28_48x44')

if not (SRC and os.path.isfile(os.path.join(SRC, 'input.cfg'))):
    import sys
    print('SKIP: set STIC_EMUL_TEST_DATA to a coupled run dir with input.cfg')
    sys.exit(0)
MODEL_N = 140
MX0, MY0, TNX, TNY = 36, 28, 48, 44

shutil.rmtree(OUT, ignore_errors=True)
rng = np.random.default_rng(11)

os.makedirs(os.path.join(HERE, '.test_scratch'), exist_ok=True)
meta = tiles.extract_tile_rundir(SRC, OUT, MX0, MY0, TNX, TNY,
                                 require_aux=False)
print('tile meta:', {k: v for k, v in meta['regions'].items()})

img_full = rng.normal(1.0, 0.3, (MODEL_N, MODEL_N))
# make it smooth-ish so numbers are O(1)
from scipy.ndimage import gaussian_filter
img_full = gaussian_filter(img_full, 2.0)
img_tile = img_full[MY0:MY0 + TNY, MX0:MX0 + TNX]

for reg in ('obs_8542.nc', 'obs_6302.nc', 'obs_3934.nc'):
    D_full = dg.build_region_operator(os.path.join(SRC, reg),
                                      MODEL_N, MODEL_N)
    D_tile = dg.build_region_operator(os.path.join(OUT, reg), TNX, TNY)

    ox0, ox1, oy0, oy1 = meta['regions'][reg]['window']
    ny_t, nx_t = meta['regions'][reg]['ny'], meta['regions'][reg]['nx']

    y_full = (D_full @ img_full.ravel()).reshape(-1)
    with np.errstate(all='ignore'):
        y_tile = (D_tile @ img_tile.ravel()).reshape(ny_t, nx_t)

    # full-op output restricted to the tile's observed window
    nyF = int(np.sqrt(D_full.shape[0])) if reg != 'obs_6302.nc' else None
    # derive full output dims from operator shape and window
    obs = tiles.CoupledObs(os.path.join(SRC, reg))
    y_full = y_full.reshape(obs.ny, obs.nx)[oy0:oy1, ox0:ox1]

    # The load-bearing property: every obs pixel that PARTICIPATES in the
    # tile inversion (pixel_weight > 0) must have an operator row
    # identical to the full-FOV one.
    tob = tiles.CoupledObs(os.path.join(OUT, reg))
    if reg == 'obs_3934.nc':
        # fine-grid region: interior = PSF footprint inside tile; here
        # the operator is identity, so everything must match
        active = np.ones((ny_t, nx_t), bool)
    else:
        active = tob.pweights[0] > 0

    err = np.abs(y_tile - y_full)[active]
    print(f'{reg}: obs window x[{ox0}:{ox1}] y[{oy0}:{oy1}] '
          f'tile obs {ny_t}x{nx_t}, active px {active.sum()}/{active.size}, '
          f'max err on active = {err.max():.3e}')
    assert active.sum() > 0.4 * active.size, f'{reg}: too few active pixels'
    assert err.max() < 1e-12, reg

print('\nALL TILE TESTS PASSED')
