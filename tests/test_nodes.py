"""
Node-based target representation: profiles built with STiC's rule
(linear between nodes, constant beyond) must be recovered exactly from
the node positions STiC derives from input.cfg. No data needed.

  python tests/test_nodes.py
"""

import numpy as np

from emulator import tiles, nn_dataset as nd, emu_config as C

rng = np.random.default_rng(5)
ltau = np.round(np.arange(-8.0, 1.0001, 0.1), 10)       # 91 points

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
print('stic_nodes:', {v: np.round(x, 2).tolist() for v, x in nodes.items()})

npix = 50
truth = {}
for v, x in nodes.items():
    lo, hi = {'temp': (3500, 20000), 'vlos': (-8e5, 8e5),
              'vturb': (0, 3e5), 'blong': (-1500, 1500),
              'bhor': (0, 1500), 'azi': (0, np.pi)}[v]
    vals = rng.uniform(lo, hi, (npix, len(x)))
    truth[v] = np.array([np.interp(ltau, x, r) for r in vals])
arrs = {v: a.reshape(1, npix, -1) for v, a in truth.items()}

for b_perp in (True, False):
    C.B_PERP_VECTOR = b_perp
    pos, exact = nd.target_positions(nodes)
    Y, names, where = nd.encode_targets_arrays(ltau, arrs, pos, exact)
    back = nd.stratify(Y, names, where, ltau)
    errs = {}
    for v in truth:
        d = back[v] - truth[v]
        if v == 'azi':
            d = (d + np.pi / 2) % np.pi - np.pi / 2
        errs[v] = np.abs(d).max() / max(np.abs(truth[v]).max(), 1e-30)
    print(f'B_PERP_VECTOR={b_perp}: {Y.shape[1]} outputs, max relative '
          f'rebuild error ' + ', '.join(f'{v} {e:.1e}' for v, e in errs.items()))
    assert max(errs.values()) < 1e-9, errs

# fixed-grid representation (old checkpoints) still decodes
C.B_PERP_VECTOR = False
pos, exact = nd.target_positions(None)
Y, names, where = nd.encode_targets_arrays(ltau, arrs, pos, exact)
back = nd.stratify(Y, names, where, ltau)
print(f'grid representation: {Y.shape[1]} outputs, rebuild works '
      f'(T error {np.abs(back["temp"] - truth["temp"]).max():.0f} K, '
      f'expected nonzero)')
print('\nALL NODE TESTS PASSED')
