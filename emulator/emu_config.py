"""Configuration for the coupled-STiC NN emulator network."""

import numpy as np

# ---- targets -------------------------------------------------------------- #
# 'nodes': the network predicts each quantity at the inversion's own STiC
#   nodes (from input.cfg, stored in the label records at harvest) and the
#   stratification is rebuilt exactly like STiC builds it (linear between
#   nodes, constant beyond), so kinks are reproduced exactly.
# 'grid': all quantities on the fixed LTAU_GRID (the original setup).
TARGET_REPR = 'nodes'
# For label records harvested before node positions were stored: the
# input.cfg of the inversion that produced them (None: re-harvest instead).
NODES_CFG = None
LTAU_GRID = np.linspace(-7.0, 0.8, 16)   # 'grid' representation / fallback
# Smallest spread used to standardize each output (target units, velocities
# in cm/s): outputs that barely vary across the training pixels, e.g. a node
# the inversion left at its starting value, would otherwise be divided by ~0.
# Differences below these are physically negligible.
TARGET_SPREAD_FLOOR = {'log_temp': 0.005, 'temp': 50.0,
                       'vlos': 1e4, 'vturb': 1e4,
                       'blong': 10.0, 'bhor': 10.0,
                       'bperp_c': 10.0, 'bperp_s': 10.0,
                       'azi_sin': 0.05, 'azi_cos': 0.05, 'azi': 0.05}
TARGET_VARS = ['temp', 'vlos', 'vturb', 'blong', 'bhor', 'azi']
LOG_TEMP = True
AZI_SINCOS = True          # used when B_PERP_VECTOR is False
# Predict the transverse field as the vector (B_hor cos 2phi, B_hor sin 2phi)
# instead of B_hor and the angle: no undefined azimuth where B_hor is weak,
# and azimuth errors count in proportion to B_hor.
B_PERP_VECTOR = True
# Stokes Q, U, V enter the network divided by the pixel's mean Stokes I
# (per spectral region). The obs files hold all four Stokes parameters in
# the same absolute units, so the same field gives weaker Q, U, V in dark
# areas; the ratio relates more directly to B.
POL_OVER_I = True

# ---- network -------------------------------------------------------------- #
ENC_CHANNELS = 64          # feature channels of each region encoder
ENC_LAYERS = 3
DEC_HIDDEN = [512, 512, 512]
DROPOUT = 0.05
N_ENSEMBLE = 4

# ---- training ------------------------------------------------------------- #
LR = 3e-4
WEIGHT_DECAY = 1e-4
STEPS = 20000              # optimizer steps (tiles per step: TILES_PER_STEP)
TILES_PER_STEP = 4         # gradient accumulation over tiles
# Spatial hold-out for validation: the fine grid of each FOV is cut into
# VAL_BLOCK x VAL_BLOCK pixel blocks, and a fixed pseudo-random
# VAL_FRACTION of the blocks is held out from training — in ALL label
# records of that FOV (full map and overlapping tiles alike). Training
# pixels within VAL_GUARD pixels of a held-out block are dropped as well,
# so strongly correlated neighbours cannot leak into the validation score.
VAL_FRACTION = 0.15
VAL_BLOCK = 20
VAL_GUARD = 3
VAL_SEED = 1234            # choose a different hold-out pattern without
                           # changing network initialization (SEED)
PATIENCE_STEPS = 3000
# Checkpoint selection / early stopping on the held-out pixels:
# 'mse'  error of the predicted values (standardized targets)
# 'nll'  Gaussian likelihood, which also scores the predicted error bars and
#        rises as soon as the networks become overconfident
EARLY_STOP_ON = 'mse'
DEPTH_SMOOTH_LAMBDA = 1e-3
CHI2_MAX = None            # drop label tiles with any region chi2 above this
# Per-PIXEL label filter for full-map records (harvest_rundir); this is
# what makes a PARTIALLY converged map usable as a label source. The
# pixel chi2 is in STiC's own units (mean of ((obs-syn)/weights)^2, worst
# region), but STiC weights are often relative rather than true noise, so
# typical values can be >> 1. Hence the default is RELATIVE: keep the
# best PIXEL_CHI2_QUANTILE fraction of each map. Setting PIXEL_CHI2_MAX
# (absolute) overrides the quantile.
PIXEL_CHI2_QUANTILE = 0.5
PIXEL_CHI2_MAX = None
MAX_PIX_PER_TILE = 2048    # decoder pixels per record per training step
DEVICE = 'auto'            # 'auto' (cuda > mps > cpu) | 'cuda' | 'mps' | 'cpu'
SEED = 1234

# ---- prediction ----------------------------------------------------------- #
PREDICT_CHUNK = 8192       # fine pixels per decoder batch
