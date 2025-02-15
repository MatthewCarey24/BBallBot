"""Module for model training and optimization in BBallBot"""

import numpy as np
import pandas as pd
import joblib
import pickle
import os
from typing import Dict, Any, Tuple
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
    # Create hidden layer structure
    hidden_layers = []
    for i in range(params['n_hidden_layers']):
        hidden_layers.append(params[f'layer_{i+1}_neurons'])
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layers),
            activation=params['activation'],
            solver=params['solver'],
            alpha=params['alpha'],
            learning_rate=params['learning_rate'],
            learning_rate_init=params['learning_rate_init'],
            max_iter=params['max_iter'],
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
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

def objective(
    trial: optuna.Trial,
    df_path: str,
    frac_test: float
) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        df_path: Path to the data file
        frac_test: Fraction of data to use for testing
        
    Returns:
        Cross-validation accuracy score
    """
    # Define hyperparameters to optimize
    params = {
        'nmf_n_components': trial.suggest_int('nmf_n_components', 5, 20),
        'alpha_H': trial.suggest_float('alpha_H', 0.0001, 0.5, log=True),
        'alpha_W': trial.suggest_float('alpha_W', 0.0001, 0.5, log=True),
        'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 3),
        'activation': trial.suggest_categorical('activation', ['tanh', 'relu', 'logistic']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd', 'lbfgs']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive', 'invscaling']),
        'max_iter': trial.suggest_int('max_iter', 1000, 5000),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.1, log=True),
    }
    
    # Add layer-specific neuron counts
    for i in range(params['n_hidden_layers']):
        params[f'layer_{i+1}_neurons'] = trial.suggest_int(f'layer_{i+1}_neurons', 5, 100)
    
    # Create features
    df = pd.read_csv(df_path)
    X, y = create_features(
        df,
        frac_test,
        params['nmf_n_components'],
        params['alpha_H'],
        params['alpha_W']
    )
    
    # Save trial data
    save_trial_data(trial.number, X, y)
    
    # Split data
    X_train, X_test = split_into_train_and_test(X, frac_test, random_state=1)
    y_train, y_test = split_into_train_and_test(y.reshape(-1, 1), frac_test, random_state=1)
    
    # Create and train model
    model = create_model(params)
    model.fit(X_train, y_train.ravel())
    
    # Evaluate
    y_pred = model.predict(X_test)
    return float(accuracy_score(y_test, y_pred))

def train_and_evaluate(
    df_path: str,
    year: int,
    frac_test: float,
    n_trials: int,
    starting_wealth: float
) -> Tuple[float, float, float]:
    """
    Train model using Optuna and evaluate its performance.
    
    Args:
        df_path: Path to the data file
        year: Year for file naming
        frac_test: Fraction of data to use for testing
        n_trials: Number of Optuna trials
        starting_wealth: Initial wealth for profit calculation
        
    Returns:
        Tuple of (accuracy, profit, profit_percentage)
    """
    # Create and run Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, df_path, frac_test),
        n_trials=n_trials
    )
    
    best_trial = study.best_trial
    print(f"Best trial: Value={best_trial.value}, Params={best_trial.params}")
    
    # Save best trial
    save_best_trial(best_trial, year)
    
    # Load best model and data
    best_classifier = joblib.load(MODEL_FILENAME.format(year=year))
    X = np.load(TRIAL_X_FILENAME.format(trial_id=best_trial.number))
    y = np.load(TRIAL_Y_FILENAME.format(trial_id=best_trial.number))
    
    # Clean up trial files
    cleanup_trial_files(best_trial.number)
    
    # Split and evaluate
    X_train, X_test = split_into_train_and_test(X, frac_test, random_state=1)
    y_train, y_test = split_into_train_and_test(y.reshape(-1, 1), frac_test, random_state=1)
    
    best_classifier.fit(X_train, y_train.ravel())
    y_pred = best_classifier.predict(X_test)
    y_proba = best_classifier.predict_proba(X_test)
    
    accuracy = float(accuracy_score(y_test, y_pred))
    wealth, total_stake = test_profit(df_path, y_pred, y_test, y_proba, starting_wealth, frac_test)
    profit = float(wealth - starting_wealth)
    profit_percentage = float(profit / total_stake if total_stake > 0 else 0)
    
    return accuracy, profit, profit_percentage
