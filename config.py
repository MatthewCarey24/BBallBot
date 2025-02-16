"""Configuration constants for BBallBot"""

# File paths
ODDS_DATA_PATH = "odds_data/odds_data_{year}.csv"

# Model parameters
RANDOM_STATE = 42  # Used for NMF initialization and MLPClassifier, NOT for train/test splitting
FRAC_TEST = 0.2  # Fraction of season to use for testing (uses latter portion of season)
STARTING_WEALTH = 1000  # Initial bankroll for betting simulation

# NMF parameters
NMF_MAX_ITER = 20000  # Maximum iterations for NMF convergence
NMF_INIT = 'nndsvdar'  # Initialization method for NMF

# Optuna study parameters
N_TRIALS = 50  # Number of trials for hyperparameter optimization

# Model file patterns
MODEL_FILENAME = "best_mlp_model_{year}.pkl"  # Format: best_mlp_model_2022.pkl
TRIAL_FILENAME = "best_trial_{year}.pkl"  # Format: best_trial_2022.pkl
PARAMS_FILENAME = "best_trial_params_{year}.pkl"  # Format: best_trial_params_2022.pkl
TRIAL_X_FILENAME = "x_trial_{trial_id}.npy"  # Format: x_trial_0.npy
TRIAL_Y_FILENAME = "y_trial_{trial_id}.npy"  # Format: y_trial_0.npy
