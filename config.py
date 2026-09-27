# -*- coding: utf-8 -*-
"""
config.py — central configuration for the SEIR PINN pipeline.

Holds paths, device setup, and every hyperparameter previously scattered
through the monolithic script's "0. KAGGLE INPUT / OUTPUT SETUP" and
"0. CONFIG" sections. All other modules (`data_preprocessing`, `models`,
`training`, `rk4_simulator`) do `import config` (not `from config import X`)
and read attributes off the module at call time — this matters specifically
for `N_DOMAINS` and `T_DAYS` below, which are data-dependent and unknown
until `data_preprocessing.load_and_preprocess_data()` has actually run.
`main.py` sets `config.T_DAYS` / `config.N_DOMAINS` right after loading data
and before any training call; every downstream `config.N_DOMAINS` lookup
resolves dynamically, so this works correctly even though config.py itself
can't know these two values at import time.

For the full modification history (mods #1-#16) that produced the current
behavior encoded here, see the module docstring in main.py — it is kept in
one place rather than duplicated across files.
"""

import os
import torch

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = "/kaggle/input/datasets/kanthamgowda/india-processed-data"
INPUT_CSV = os.path.join(BASE_DIR, "india_processed_data.csv")
OUTPUT_BASE = "/kaggle/working/"  # checkpoints + processed CSVs are written here

os.makedirs(BASE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# MULTI-GPU SETUP
# ─────────────────────────────────────────────────────────────
NUM_GPUS = torch.cuda.device_count()
if NUM_GPUS >= 2:
    DEVICES = [torch.device(f'cuda:{i}') for i in range(NUM_GPUS)]
elif NUM_GPUS == 1:
    DEVICES = [torch.device('cuda:0')]
else:
    DEVICES = [torch.device('cpu')]
device = DEVICES[0]

# ─────────────────────────────────────────────────────────────
# COLUMN / FEATURE SCHEMA
# ─────────────────────────────────────────────────────────────
DATA_COLUMNS = {
    'time': 't',
    'x': 'x',
    'y': 'y',
    'targets': ['S', 'I', 'R'],
    'param_features': [
        'x', 'y', 'stringency_smooth', 'pop_density',
        'velocity', 'acceleration', 'regime_signal', 'spike_signal',
    ],
}

# Which param_features are genuinely exogenous (safe to read from the true
# series at any query time during the RK4 rollout) vs. trajectory-derived
# (must instead be produced online from the simulated state — mod #14).
EXO_PARAM_FEATURES = ['x', 'y', 'stringency_smooth', 'pop_density']
TRAJ_PARAM_FEATURES = ['velocity', 'acceleration', 'regime_signal', 'spike_signal']

# Target column feature engineering is built from (calculate_epidemic_dynamics
# operates on this column's rate of change).
TARGET_COL = 'I'

# ─────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# ─────────────────────────────────────────────────────────────
# DOMAIN DECOMPOSITION — data-dependent, set by main.py after
# data_preprocessing.load_and_preprocess_data() runs. See module docstring.
# ─────────────────────────────────────────────────────────────
SUBDOMAIN_LENGTH_DAYS = 30
T_DAYS = None       # set by main.py
N_DOMAINS = None    # set by main.py
OVERLAP_FRAC = 0.10

# ─────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
EPOCHS_PER_DOMAIN = 7000
WARMUP_EPOCHS = 1000
N_COLLOC_MAX = 8000
LR = 1e-3
# Separate, higher learning rate for the ParamNet parameter group. Each
# per-domain ParamNet only receives gradient through the long, thin
# ODE-residual path, while each per-domain StateNet gets large, direct
# gradients from the 1e5-weighted data/boundary MSE terms. Giving the
# pooled ParamNet parameters their own higher LR (and their own
# clip_grad_norm_ call, applied in training.py) means that gradient signal
# isn't diluted by sharing an LR/clip budget tuned for the much larger
# StateNet gradients.
PARAM_NET_LR = 1e-2
MIN_ROWS_PER_DISTRICT = 50

# How often (in epochs) to write checkpoints to disk and pull loss values
# back to Python for logging. Doing this every epoch was a major source of
# slowness: save_checkpoint() was hitting disk N_DOMAINS times per epoch
# (thousands of small writes over a full run), and calling .item() on every
# loss component for every domain every epoch forced a GPU/CPU sync
# ~5*N_DOMAINS times per epoch, serializing the loop.
CHECKPOINT_INTERVAL = 500

TARGET_DISTRICT = "IN_KL_ERN"

SCALES = {'S': 1.0, 'E': 1.0, 'I': 1.0, 'R': 1.0}
INIT_WEIGHTS = {'data': 1e5, 'ode': 0.0, 'alg': 0.0, 'boundary': 1e5}

# mod #11 — ode/alg weight ramp target/duration.
ODE_ALG_TARGET_WEIGHT = 1e5
ODE_ALG_RAMP_EPOCHS = 1000

# mod #15 — ParamNet Fourier features. Applies only to ParamNet's raw
# contextual input; StateNet's (t, x, y) input is untouched.
PARAM_NET_FOURIER_NUM_FREQS = 16
PARAM_NET_FOURIER_SCALE = 4.0

# ─────────────────────────────────────────────────────────────
# COLLOCATION SAMPLING
# ─────────────────────────────────────────────────────────────
COLLOC_CURVATURE_FRAC = 0.7
COLLOC_JITTER_STD = 0.05

# ─────────────────────────────────────────────────────────────
# RK4 ROLLOUT
# ─────────────────────────────────────────────────────────────
RK4_STEPS_PER_INTERVAL = 20
