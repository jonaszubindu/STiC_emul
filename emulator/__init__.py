"""STiC_emul: NN emulator for (spatially coupled) STiC inversions.

The emulator interacts with coupled STiC only through files and a
subprocess boundary — it needs a compiled STiC.x binary (any machine),
a template run directory, and nothing else from the C++ side.

Modules:
  stic_io      STiC model/profile netCDF I/O (vendored, self-contained)
  degradation  verified sparse spatial degradation operators D_r
  tiles        tile run-dir extraction with exact operator re-basing
  labels       tile sampling, batch drivers (local/SLURM), harvesting,
               full-map harvesting from partially converged inversions
  nn_dataset   tile records -> training samples / prediction inputs
  net          multi-res encoders + LT-aware LIIF decoder
  train_net    ensemble training (per-GPU members via --member)
  predict_map  full-map prediction + Nyquist checkerboard diagnostic
"""

__version__ = '0.1.0'
