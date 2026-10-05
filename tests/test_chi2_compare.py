"""
chi2 comparison: degrading a coupled run's own fine-grid synthetic
(output_profiles) must reproduce the chi2 of STiC's out_<obs> files, i.e.
an emulator/inversion ratio of 1 in every region.

  STIC_EMUL_TEST_DATA=<coupled run dir with out_obs_*.nc and the
  output_profiles file> python tests/test_chi2_compare.py
"""

import os
import sys
import tempfile

from emulator import chi2_compare as cc, tiles

run = os.environ.get('STIC_EMUL_TEST_DATA')
if not run or not os.path.isfile(os.path.join(run, 'input.cfg')):
    print('SKIP: set STIC_EMUL_TEST_DATA to a coupled run dir with '
          'out_obs_*.nc and its output_profiles synthetic')
    sys.exit(0)
with open(os.path.join(run, 'input.cfg')) as f:
    synth = os.path.join(run, tiles.read_cfg_last(f.read(), 'output_profiles'))

with tempfile.TemporaryDirectory() as d:
    cc.degrade(run, synth, os.path.join(d, 'emu'))
    s = cc.compare(run, run, os.path.join(d, 'emu'), os.path.join(d, 'cmp'))
for tag, r in s['regions'].items():
    assert abs(r['ratio_median'] - 1) < 1e-4, (tag, r)
    assert r['frac_emu_as_good'] > 0.99, (tag, r)
print('\nCHI2 COMPARISON TEST PASSED')
