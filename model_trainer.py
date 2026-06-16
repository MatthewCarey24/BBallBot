"""Module for model training and optimization in BBallBot"""

import os
import pickle
import random
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from betting import simulate_bankroll, test_profit
from config import (
    EDGE_THRESHOLD,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    MODEL_FILENAME,
    PARAMS_FILENAME,
    RANDOM_STATE,
    STARTING_WEALTH,
    TRIAL_FILENAME,
    TRIAL_X_FILENAME,
    TRIAL_Y_FILENAME,
    USE_ODDS_FEATURE,
)
from features.combined import build_feature_matrix
from features.ratings import EloSystem


def fit_calibrated(params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, calib_frac: float = 0.2, random_state: int = RANDOM_STATE) -> Any:
    """Fit an MLP and calibrate it on a *held-out* tail of the training data.

    The base model is trained on the first ``1 - calib_frac`` of the training
    rows; the remaining tail (chronologically last, never seen during fitting)
    is used to fit the isotonic calibrator.  Calibrating on held-out data —
    rather than the data the model already fit — keeps the probabilities honest,
    which matters because the betting layer bets on model-vs-market edge.

    Returns a fitted ``CalibratedClassifierCV``.
    """
    n = len(X_train)
    n_calib = max(1, int(n * calib_frac))
    X_fit, y_fit = X_train[:-n_calib], y_train[:-n_calib]
    X_cal, y_cal = X_train[-n_calib:], y_train[-n_calib:]

    base = create_model(params, random_state=random_state)
    base.fit(X_fit, y_fit.ravel())

    # sklearn >= 1.6 removed cv='prefit' in favour of FrozenEstimator; fall
    # back to cv='prefit' on older versions (supports scikit-learn >= 1.3).
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method='isotonic')
    except ImportError:
        calibrated = CalibratedClassifierCV(base, method='isotonic', cv='prefit')
    calibrated.fit(X_cal, y_cal.ravel())
    return calibrated


def create_model(params: Dict[str, Any], random_state: int = RANDOM_STATE) -> Pipeline:
    """
    Create a pipeline with standardization and MLP classifier.

    Args:
        params: Dictionary of model parameters
        random_state: Seed for the MLP (override to study seed variance).

    Returns:
        Sklearn Pipeline object
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(params['first_layer_neurons'],),
            activation=params['activation'],
            solver=params['solver'],
            alpha=params['alpha'],
            learning_rate=params['learning_rate'],
            learning_rate_init=params['learning_rate_init'],
            max_iter=20000,
            random_state=random_state
        ))
    ])

def save_trial_data(trial_id: int, X: np.ndarray, y: np.ndarray) -> None:
    """
    Save trial data to files.

    Args:
        trial_id: ID of the trial
        X: Feature matrix
        y: Target vector
    """
    np.save(TRIAL_X_FILENAME.format(trial_id=trial_id), X)
    np.save(TRIAL_Y_FILENAME.format(trial_id=trial_id), y)

def save_best_trial(best_trial: optuna.trial.FrozenTrial, year: int) -> None:
    """
    Save the best trial's model and parameters.

    Args:
        best_trial: Best trial from Optuna study
        year: Year for file naming
    """
    # Save the model
    best_classifier = create_model(best_trial.params)
    model_path = MODEL_FILENAME.format(year=year)
    joblib.dump(best_classifier, model_path)
    print(f"Model saved to {model_path}")

    # Save the trial
    trial_path = TRIAL_FILENAME.format(year=year)
    with open(trial_path, 'wb') as f:
        pickle.dump(best_trial, f)
    print(f"Best trial saved to {trial_path}")

    # Save parameters
    params_path = PARAMS_FILENAME.format(year=year)
    with open(params_path, 'wb') as f:
        pickle.dump(best_trial.params, f)
    print(f"Best trial parameters saved to {params_path}")

def cleanup_trial_files(best_trial_id: int) -> None:
    """
    Remove trial data files except for the best trial.

    Args:
        best_trial_id: ID of the best trial to keep
    """
    for filename in os.listdir('.'):
        if filename.startswith('x_trial_') or filename.startswith('y_trial_'):
            trial_number = filename.split('_')[2].split('.')[0]
            if int(trial_number) != best_trial_id:
                os.remove(os.path.join('.', filename))

def set_random_seeds(seed: int = RANDOM_STATE) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def objective(
    trial: optuna.Trial,
    df_path: str,
    frac_test: float,
    use_odds: bool = USE_ODDS_FEATURE,
) -> float:
    """
    Objective function for Optuna optimization using only training data.

    Scores hyperparameters by the *betting ROI* of the independent model on a
    held-out validation slice — not raw accuracy.  Because odds are excluded
    from the features (``USE_ODDS_FEATURE``), the model forms an independent
    estimate, and we reward configurations whose disagreements with the
    vig-free market actually turn a profit rather than ones that merely
    reproduce the favourite.

    Args:
        trial: Optuna trial object
        df_path: Path to the data file
        frac_test: Fraction of data to use for testing

    Returns:
        Validation return-on-bankroll (profit / starting wealth).
    """
    # Set random seeds for reproducibility
    set_random_seeds()

    # Define hyperparameters to optimize
    params = {
        'nmf_n_components': trial.suggest_int('nmf_n_components', 3, 10),
        'alpha_H': trial.suggest_float('alpha_H', 0.0001, 0.1001, step=0.005),
        'alpha_W': trial.suggest_float('alpha_W', 0.0001, 0.1001, step=0.005),
        'first_layer_neurons': trial.suggest_int('first_layer_neurons', 5, 50),
        'activation': trial.suggest_categorical('activation', ['tanh', 'relu']),
        'solver': trial.suggest_categorical('solver', ['sgd', 'adam']),
        'alpha': trial.suggest_float('alpha', 1e-3, 1e1, log=True),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.1001, log=True),
    }

    # Read data and get training portion only
    df = pd.read_csv(df_path)
    train_size = int(len(df) * (1 - frac_test))
    df_train = df.iloc[:train_size].reset_index(drop=True)

    # Create features (NMF + Elo + form, no odds) using only training data.
    # A fresh EloSystem is used so ratings only learn from training games.
    X, y = build_feature_matrix(
        df_train,  # Only use training portion for feature creation
        EloSystem(),
        0.2,  # Use 20% of training data for validation (NMF/Elo update boundary)
        params['nmf_n_components'],
        params['alpha_H'],
        params['alpha_W'],
        use_odds=use_odds,
    )

    # Chronological split of the training rows into:
    #   fit (model) | calibration | validation (betting score)
    val_size = int(len(X) * 0.2)
    X_trainval, X_val = X[:-val_size], X[-val_size:]
    y_trainval, y_val = y[:-val_size], y[-val_size:]
    val_odds = df_train.iloc[-val_size:]

    # Fit + held-out calibration so the validation probabilities are honest
    model = fit_calibrated(params, X_trainval, y_trainval)

    # Score by betting ROI against the vig-free market on the validation slice
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    wealth, _ = simulate_bankroll(
        val_odds, y_pred, y_val, y_proba, STARTING_WEALTH,
        kelly_fraction=KELLY_FRACTION,
        max_fraction=MAX_BET_FRACTION,
        edge_threshold=EDGE_THRESHOLD,
    )
    return float((wealth - STARTING_WEALTH) / STARTING_WEALTH)

def train_and_evaluate(
    df_path: str,
    year: int,
    frac_test: float,
    n_trials: int,
    starting_wealth: float,
    use_saved_params: bool = False
) -> Tuple[float, float, float, float]:
    """
    Train model using Optuna and evaluate its performance.

    Args:
        df_path: Path to the data file
        year: Year for file naming
        frac_test: Fraction of data to use for testing
        n_trials: Number of Optuna trials
        starting_wealth: Initial wealth for profit calculation
        use_saved_params: If True, use previously saved parameters instead of optimizing

    Returns:
        Tuple of (accuracy, profit, profit_percentage, brier_score)
    """
    # Set random seeds for reproducibility
    set_random_seeds()

    params_path = PARAMS_FILENAME.format(year=year)

    if use_saved_params and os.path.exists(params_path):
        # Load previously saved parameters
        with open(params_path, 'rb') as f:
            best_params = pickle.load(f)
        print(f"Using saved parameters from {params_path}")

        # Create a dummy trial to store parameters
        best_trial = optuna.trial.create_trial(
            params=best_params,
            distributions={},
            value=0.0  # Placeholder value
        )
    else:
        # Create and run Optuna study with fixed random seed for reproducibility
        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)  # Use the same random state
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(
            lambda trial: objective(trial, df_path, frac_test),
            n_trials=n_trials
        )
        best_trial = study.best_trial

    print(f"Best trial: Value={best_trial.value}, Params={best_trial.params}")

    # Build the full feature matrix once (NMF + Elo + form), then split.
    # NMF fits and Elo updates only on the training portion (leakage-safe);
    # form features are pre-game by construction.
    df = pd.read_csv(df_path)
    train_size = int(len(df) * (1 - frac_test))

    X, y = build_feature_matrix(
        df,
        EloSystem(),
        frac_test,
        best_trial.params['nmf_n_components'],
        best_trial.params['alpha_H'],
        best_trial.params['alpha_W'],
        use_odds=USE_ODDS_FEATURE,
    )

    X_train = np.array(X[:train_size])
    y_train = np.array(y[:train_size])
    X_test = np.array(X[train_size:])
    y_test = np.array(y[train_size:])

    # Persist the feature matrix for reproducibility/inspection
    save_trial_data(0, X, y)

    # Train final model with best params, calibrated on a held-out tail of the
    # training data (honest probabilities for the value-betting layer).
    calibrated_model = fit_calibrated(best_trial.params, X_train, y_train)

    # Save model and trial info
    save_best_trial(best_trial, year)

    # Evaluate on test set using calibrated model
    y_pred = np.array(calibrated_model.predict(X_test))
    y_proba = np.array(calibrated_model.predict_proba(X_test))

    accuracy = float(accuracy_score(y_test, y_pred))
    brier = float(brier_score_loss(y_test, y_proba[:, 1]))
    print(f"Brier Score: {brier:.4f}")

    wealth, total_stake = test_profit(
        df_path, y_pred, y_test, y_proba, starting_wealth, frac_test,
        kelly_fraction=KELLY_FRACTION,
        max_fraction=MAX_BET_FRACTION,
        edge_threshold=EDGE_THRESHOLD,
    )
    profit = float(wealth - starting_wealth)
    profit_percentage = float(profit / total_stake if total_stake > 0 else 0)

    return accuracy, profit, profit_percentage, brier
