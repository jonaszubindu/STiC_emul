"""Configuration for the coupled-STiC NN emulator network."""

import numpy as np

# ---- targets -------------------------------------------------------------- #
LTAU_GRID = np.linspace(-7.0, 0.8, 16)
TARGET_VARS = ['temp', 'vlos', 'vturb', 'blong', 'bhor', 'azi']
LOG_TEMP = True
AZI_SINCOS = True

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
