"""
Node-based target representation: atmospheres built like coupled STiC
writes them (expandAtmos: linear between nodes, linearly continued beyond
the outermost ones, only the quantities inverted in a cycle are rebuilt)
must be recovered exactly from the node positions STiC derives from the
input.cfg of each cycle. No data needed.

  python tests/test_nodes.py
"""

import numpy as np

from emulator import tiles, nn_dataset as nd, emu_config as C

rng = np.random.default_rng(5)
ltau = np.round(np.arange(-8.0, 1.0001, 0.1), 10)       # 91 points


def linpol_extrap(xx, x, y):
    """Port of STiC's linpol(..., extrapolate=true) (interpol.h)."""
    yy = np.empty(len(xx))
    for k in range(1, len(x)):
        a = (y[k] - y[k - 1]) / (x[k] - x[k - 1])
        sel = (xx >= x[k - 1]) & (xx < x[k])
        yy[sel] = a * xx[sel] + y[k - 1] - a * x[k - 1]
    a0 = (y[1] - y[0]) / (x[1] - x[0])
    a1 = (y[-1] - y[-2]) / (x[-1] - x[-2])
    lo, hi = xx <= x[0], xx >= x[-1]
    yy[lo] = a0 * xx[lo] + y[0] - a0 * x[0]
    yy[hi] = a1 * xx[hi] + y[-1] - a1 * x[-1]
    return yy


def stic_expand(x, y):
    """mdepthall::expand with depth_interpolation = 0."""
    return np.full(ltau.size, y[0]) if len(x) == 1 \
        else linpol_extrap(ltau, x, y)


# --- node rules -------------------------------------------------------------
cfg = """
depth_t = 0
depth_interpolation = 0
nodes_temp = 5
nodes_vlos = 3
nodes_vlos = -9., -5.3, -4.6, -3.9, -3.2, -2.5, -1.8, -1.1, -0.4, 0.3, 1.
nodes_vturb = 2
nodes_blong = 3
nodes_bhor = 2
nodes_azi = 1   # one node: constant azimuth
"""
nodes = tiles.stic_nodes(cfg, ltau)
assert len(nodes['vlos']) == 11, 'last assignment of nodes_vlos must win'
assert nodes['vlos'][0] == -8.0, 'explicit -9 snaps to the grid edge'
# inner equidistant nodes snap to the grid (first match on ties)
assert np.allclose(nodes['temp'], [-8, -5.8, -3.5, -1.3, 1.0]), nodes['temp']
assert all(np.isclose(ltau, x).any() for x in nodes['temp'])

# --- two cycles: everything with T/vlos nodes ending at 0.8 (inside the
# grid), then a magnetic-field-only cycle -------------------------------------
cycle1 = """
depth_t = 0
depth_interpolation = 0
nodes_temp = -8.0, -6.0, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.0, 0.0, 0.8
nodes_vlos = -8.0, -6.0, -5.0, -3.0, -1.0, 0.0, 0.8
nodes_vturb = 3
nodes_blong = 2
nodes_bhor = 2
nodes_azi = 1
"""
cycle2 = """
depth_t = 0
depth_interpolation = 0
nodes_temp = 0
nodes_vlos = 0
nodes_vturb = 0
nodes_blong = 4
nodes_bhor = 3
nodes_azi = 1
"""
last_only = tiles.stic_nodes_cycles([cycle2], ltau)
assert len(last_only['temp']) == 0
nodes = tiles.stic_nodes_cycles([cycle1, cycle2], ltau)
assert nodes['temp'][-1] == 0.8 and len(nodes['temp']) == 12
assert len(nodes['vturb']) == 3, 'vturb from cycle 1'
assert np.allclose(nodes['blong'], [-8, -5, -2, 1]), 'B from cycle 2'
print('stic_nodes_cycles:', {v: np.round(x, 2).tolist()
                             for v, x in nodes.items()})

npix = 50
truth = {}
for v, x in nodes.items():
    lo, hi = {'temp': (3500, 20000), 'vlos': (-8e5, 8e5),
              'vturb': (0, 3e5), 'blong': (-1500, 1500),
              'bhor': (0, 1500), 'azi': (0, np.pi)}[v]
    vals = rng.uniform(lo, hi, (npix, len(x)))
    truth[v] = np.array([stic_expand(x, r) for r in vals])
assert np.abs(truth['temp'][:, -1] - truth['temp'][:, -3]).max() > 0, \
    'T must keep varying beyond the last node (linear continuation)'
arrs = {v: a.reshape(1, npix, -1) for v, a in truth.items()}

for b_perp in (True, False):
    C.B_PERP_VECTOR = b_perp
    pos, exact = nd.target_positions(nodes)
    Y, names, where = nd.encode_targets_arrays(ltau, arrs, pos, exact)
    back = nd.stratify(Y, names, where, ltau, nd.target_extrap(names, exact))
    errs = {}
    for v in truth:
        d = back[v] - truth[v]
        if v == 'azi':
            d = (d + np.pi / 2) % np.pi - np.pi / 2
        errs[v] = np.abs(d).max() / max(np.abs(truth[v]).max(), 1e-30)
    print(f'B_PERP_VECTOR={b_perp}: {Y.shape[1]} outputs, max relative '
          f'rebuild error ' + ', '.join(f'{v} {e:.1e}' for v, e in errs.items()))
    assert max(errs.values()) < 1e-9, errs

# the representation check passes with both cycles ...
C.B_PERP_VECTOR = True
rec = dict(arrs)
pos, exact = nd.target_positions(nodes)
nd._check_representation(rec, pos, exact, ltau)
# ... and stops training with the last cycle's cfg alone (T, vlos, vturb
# would fall back to the fixed grid)
pos, exact = nd.target_positions(last_only)
try:
    nd._check_representation(rec, pos, exact, ltau)
except ValueError as e:
    assert 'temp' in str(e)
    print('last cycle only -> refused as expected')
else:
    raise AssertionError('representation check must fail for the last '
                         'cycle alone')

# fixed-grid representation (old checkpoints) still decodes
C.B_PERP_VECTOR = False
pos, exact = nd.target_positions(None)
Y, names, where = nd.encode_targets_arrays(ltau, arrs, pos, exact)
back = nd.stratify(Y, names, where, ltau)
print(f'grid representation: {Y.shape[1]} outputs, rebuild works '
      f'(T error {np.abs(back["temp"] - truth["temp"]).max():.0f} K, '
      f'expected nonzero)')
print('\nALL NODE TESTS PASSED')
