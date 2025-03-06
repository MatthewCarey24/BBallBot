"""Module for model training and optimization in BBallBot"""

import numpy as np
import pandas as pd
import joblib
import pickle
import os
from typing import Dict, Any, Tuple
import random
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import optuna

from config import (
    RANDOM_STATE,
    MODEL_FILENAME,
    TRIAL_FILENAME,
    PARAMS_FILENAME,
    TRIAL_X_FILENAME,
    TRIAL_Y_FILENAME
)
from data_processor import create_features
from betting import test_profit
from utils import split_into_train_and_test

def create_model(params: Dict[str, Any]) -> Pipeline:
    """
    Create a pipeline with standardization and MLP classifier.
    
    Args:
        params: Dictionary of model parameters
        
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
            max_iter=params['max_iter'],
            random_state=RANDOM_STATE
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
    frac_test: float
) -> float:
    """
    Objective function for Optuna optimization using only training data.
    
    Args:
        trial: Optuna trial object
        df_path: Path to the data file
        frac_test: Fraction of data to use for testing
        
    Returns:
        Validation accuracy score
    """
    # Set random seeds for reproducibility
    set_random_seeds()
    
    # Define hyperparameters to optimize
    params = {
        'nmf_n_components': trial.suggest_int('nmf_n_components', 5, 7),
        'alpha_H': trial.suggest_float('alpha_H', 0.0001, 0.1001, step=0.005),
        'alpha_W': trial.suggest_float('alpha_W', 0.0001, 0.1001, step=0.005),
        'first_layer_neurons': trial.suggest_int('first_layer_neurons', 1, 20),
        'activation': trial.suggest_categorical('activation', ['tanh', 'relu']),
        'solver': trial.suggest_categorical('solver', ['sgd', 'adam']),
        'alpha': trial.suggest_float('alpha', 1e1, 5e1, log=True),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
        'max_iter': trial.suggest_int('max_iter', 20000, 20001),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.1001, log=True),
    }
    
    # Read data and get training portion only
    df = pd.read_csv(df_path)
    train_size = int(len(df) * (1 - frac_test))
    df_train = df.iloc[:train_size]
    
    # Create features using only training data
    X, y = create_features(
        df_train,  # Only use training portion for feature creation
        0.2,  # Use 20% of training data for validation
        params['nmf_n_components'],
        params['alpha_H'],
        params['alpha_W']
    )
    
    # Split training data into train and validation
    val_size = int(len(X) * 0.2)  # 20% of training data for validation
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    
    # Create and train model
    model = create_model(params)
    model.fit(X_train, y_train.ravel())
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    return float(accuracy_score(y_val, y_pred))

def train_and_evaluate(
    df_path: str,
    year: int,
    frac_test: float,
    n_trials: int,
    starting_wealth: float,
    use_saved_params: bool = False
) -> Tuple[float, float, float]:
    # Set random seeds for reproducibility
    set_random_seeds()
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
        Tuple of (accuracy, profit, profit_percentage)
    """
    params_path = PARAMS_FILENAME.format(year=year)
    
    if use_saved_params and os.path.exists(params_path):
        # Load previously saved parameters
        with open(params_path, 'rb') as f:
            best_params = pickle.load(f)
        print(f"Using saved parameters from {params_path}")
        
        # Create features with saved parameters
        df = pd.read_csv(df_path)
        X, y = create_features(
            df,
            frac_test,
            best_params['nmf_n_components'],
            best_params['alpha_H'],
            best_params['alpha_W']
        )
        
        # Create a dummy trial to store parameters
        study = optuna.create_study(direction='maximize')
        trial = optuna.trial.create_trial(
            params=best_params,
            distributions={},
            value=0.0  # Placeholder value
        )
        best_trial = trial
        
        # Save trial data
        save_trial_data(0, X, y)
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
    
    # Get full dataset for final model
    df = pd.read_csv(df_path)
    train_size = int(len(df) * (1 - frac_test))
    df_train = df.iloc[:train_size]  # Only use training portion
    
    # Create features for final model using only training data
    X_train, y_train = create_features(
        df_train,
        0.0,  # No test split needed since we're using all training data
        best_trial.params['nmf_n_components'],
        best_trial.params['alpha_H'],
        best_trial.params['alpha_W']
    )
    
    # Create and train final model with best parameters
    best_model = create_model(best_trial.params)
    best_model.fit(X_train, y_train.ravel())
    
    # Save model and trial info
    save_best_trial(best_trial, year)
    
    # Create features for full dataset to get test portion
    features = create_features(
        df,
        frac_test,
        best_trial.params['nmf_n_components'],
        best_trial.params['alpha_H'],
        best_trial.params['alpha_W']
    )
    X, y = features  # Explicitly unpack tuple
    
    # Get test portion and ensure numpy array types
    X_test = np.array(X[train_size:])
    y_test = np.array(y[train_size:])
    
    # Evaluate on test set
    y_pred = np.array(best_model.predict(X_test))
    y_proba = np.array(best_model.predict_proba(X_test))
    
    accuracy = float(accuracy_score(y_test, y_pred))
    wealth, total_stake = test_profit(df_path, y_pred, y_test, y_proba, starting_wealth, frac_test)
    profit = float(wealth - starting_wealth)
    profit_percentage = float(profit / total_stake if total_stake > 0 else 0)
    
    return accuracy, profit, profit_percentage
