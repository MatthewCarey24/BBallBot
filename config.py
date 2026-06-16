"""Configuration constants for BBallBot"""

# File paths
ODDS_DATA_PATH = "odds_data/odds_data_{year}.csv"

# Model parameters
RANDOM_STATE = 41  # Used for NMF initialization and MLPClassifier, NOT for train/test splitting
FRAC_TEST = 0.05  # Fraction of season to use for testing (uses latter portion of season)
STARTING_WEALTH = 1000  # Initial bankroll for betting simulation

# Feature / betting strategy
USE_ODDS_FEATURE = False  # If False, the model never sees the line; odds are only a betting benchmark
KELLY_FRACTION = 0.25  # Fractional Kelly multiplier (quarter-Kelly)
MAX_BET_FRACTION = 0.05  # Hard cap on fraction of bankroll per bet
EDGE_THRESHOLD = 0.03  # Only bet when model prob exceeds vig-free market prob by this much

# NMF parameters
NMF_MAX_ITER = 20000  # Maximum iterations for NMF convergence
NMF_INIT = 'nndsvdar'  # Initialization method for NMF

# Optuna study parameters
N_TRIALS = 20  # Number of trials for hyperparameter optimization

# Model file patterns
MODEL_FILENAME = "best_mlp_model_{year}.pkl"  # Format: best_mlp_model_2022.pkl
TRIAL_FILENAME = "best_trial_{year}.pkl"  # Format: best_trial_2022.pkl
PARAMS_FILENAME = "best_trial_params_{year}.pkl"  # Format: best_trial_params_2022.pkl
TRIAL_X_FILENAME = "x_trial_{trial_id}.npy"  # Format: x_trial_0.npy
TRIAL_Y_FILENAME = "y_trial_{trial_id}.npy"  # Format: y_trial_0.npy
