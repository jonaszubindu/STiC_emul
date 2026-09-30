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
VAL_FRACTION = 0.1         # fraction of tiles held out
PATIENCE_STEPS = 3000
DEPTH_SMOOTH_LAMBDA = 1e-3
CHI2_MAX = None            # drop label tiles with any region chi2 above this
# Per-PIXEL label filter for full-map records (harvest_rundir): pixels
# whose combined chi2 exceeds this never become training labels. This is
# what makes a PARTIALLY converged map usable as a label source.
PIXEL_CHI2_MAX = 3.0
MAX_PIX_PER_TILE = 2048    # decoder pixels per record per training step
DEVICE = 'auto'            # 'auto' (cuda > mps > cpu) | 'cuda' | 'mps' | 'cpu'
SEED = 1234

# ---- prediction ----------------------------------------------------------- #
PREDICT_CHUNK = 8192       # fine pixels per decoder batch
