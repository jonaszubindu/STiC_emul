# STiC_emul

NN emulator for (spatially coupled) STiC inversions: invert a stratified
subset of tiles with STiC, train a multi-resolution neural network on the
(profiles → atmosphere) pairs, predict full maps, and verify by
synthesizing back through the (verified) degradation operators.

**Standalone by design**: this repo never links against coupled STiC. It
needs only (a) a compiled `STiC.x` on whatever machine runs inversions,
(b) one template run directory (obs/inst/model netCDFs + `input.cfg` +
`Atoms/ Molecules/ Atmos/ ...`), and (c) Python with
numpy/scipy/netCDF4 (+ torch where training happens).

```bash
pip install -e .          # inversion/harvest side (CPU cluster, laptop)
pip install -e '.[train]' # training side (GPU cluster)
```

**GPU setup:** the PyTorch build must support the GPU (H100 = sm_90 needs
a CUDA 12 build, e.g. `pip install torch --index-url
https://download.pytorch.org/whl/cu124`, chosen to match the driver's
`CUDA Version` in `nvidia-smi`). An old `pip install --user` PyTorch in
`~/.local` takes precedence over the conda env's; set
`export PYTHONNOUSERSITE=1` when installing and in job scripts.
`train_net` prints which torch it loaded and stops with a clear error if
that build cannot run on the GPU.

## Pipeline

```python
from emulator import labels

# Round 0 labels — free — from an existing (partially converged) map run:
labels.harvest_rundir('/path/to/map_run_dir', apron=8)
# (training keeps only pixels with chi2 <= emu_config.PIXEL_CHI2_MAX)

# Tile labels:
sel = labels.select_tiles([src_dir], tile=56, apron=8, n_tiles=32)
labels.prepare_runs(sel, out_root, tile=56, aux_search=(...,))
labels.write_slurm(out_root, STIC_BIN, ntasks=32)   # sbatch it
labels.harvest(out_root, apron=8)
```

```bash
# GPU cluster: 4 ensemble members, one per GPU:
bash cluster/train_members.sh <label_dir> [...] <ckpt_dir>
python -m emulator.predict_map <map_run_dir> --ckpt <ckpt_dir> --out <pred>
```

`predict_map` writes a STiC-ready `predicted_atmos.nc` (use it as
`input_model` for the next tile round — the bootstrap loop), uncertainty
maps, and `nyquist.json` (checkerboard null-space diagnostic; chi2 is
blind to it).

## Two-cluster operation

Every cross-machine hand-off is a small file transfer (labels ~tens of
MB, predictions ~tens of MB), so an SSH-only link between the CPU
cluster (SLURM label inversions) and the GPU cluster (training,
prediction) is sufficient. `cluster/round.sh` is a one-round
orchestration template to run from any machine that reaches both;
copy `site.yaml.example` to `site.yaml` per machine and adapt the
variables at the top of the scripts. Code moves by `git pull`, data by
`rsync` — never edit code on the clusters directly.

## Provenance and verification

Extracted from the development history in
`coupled_stic/emulator/` + `STIC_vers_2025/nn_emulator` (Jul 2026):

- `degradation.py` reimplements STiC's per-region spatial operators
  (PSF, area-overlap rebin, destretch warp, bin+trim; composition
  `D_r = LT_n ... LT_1 · PSF`) from the specs stored in each
  `obs_*.nc`; verified against the C++ (`tests/gt_dump.cc`, which
  includes STiC's own `linear_transformations.hpp`) and against STiC's
  runtime-degraded outputs at ~1e-14.
- `tiles.py` cuts self-contained tile run dirs whose operators are
  *identical* to the full-FOV ones for every participating pixel (dual
  model-space/warp-space coverage masking; apron from `suggest_apron`).
- `net.py`: per-region CNN encoders at native resolution (no transposed
  convolutions), LT-aware coordinate sampling through the stored
  geometry, heteroscedastic LIIF decoder, deep ensemble.

Tests (skip cleanly when data/source is absent):

```bash
export STIC_EMUL_TEST_DATA=/path/to/a/coupled_run_dir   # with input.cfg
export STIC_SRC=/path/to/coupled_stic/src               # for gt_dump
python tests/test_degradation.py
python tests/test_tiles.py
```
