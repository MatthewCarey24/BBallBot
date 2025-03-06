"""Main entry point for BBallBot"""

import os
from typing import Optional
import pandas as pd

from config import (
    ODDS_DATA_PATH,
    FRAC_TEST,
    STARTING_WEALTH,
    N_TRIALS,
    MODEL_FILENAME
)
from model_trainer import train_and_evaluate
import joblib

def train_model(year: int, use_saved_params: bool = False) -> None:
    """
    Train a new model for the specified year.
    
    Args:
        year: Year to train model for
        use_saved_params: If True, use previously saved parameters instead of optimizing
    """
    df_path = ODDS_DATA_PATH.format(year=year)
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"No odds data found for year {year}")
    
    print(f"\nTraining model for {year}...")
    if use_saved_params:
        print("Using saved parameters for consistent results")
    
    accuracy, profit, profit_percentage = train_and_evaluate(
        df_path=df_path,
        year=year,
        frac_test=FRAC_TEST,
        n_trials=N_TRIALS,
        starting_wealth=STARTING_WEALTH,
        use_saved_params=use_saved_params
    )
    
    print(f"\nResults for {year}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Profit: ${profit:.2f}")
    print(f"Profit Percentage: {profit_percentage*100:.2f}%")

def load_model(year: int) -> Optional[object]:
    """
    Load a trained model for the specified year.
    
    Args:
        year: Year to load model for
        
    Returns:
        Loaded model or None if not found
    """
    model_path = MODEL_FILENAME.format(year=year)
    if not os.path.exists(model_path):
        print(f"No trained model found for year {year}")
        return None
    
    return joblib.load(model_path)

def main() -> None:
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BBallBot - Basketball Betting Bot')
    parser.add_argument('--year', type=int, required=True, help='Year to train/evaluate model for')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--use-saved-params', action='store_true', help='Use saved parameters for consistent results')
    
    args = parser.parse_args()
    
    if args.train:
        train_model(args.year, use_saved_params=args.use_saved_params)
    else:
        model = load_model(args.year)
        if model is not None:
            print(f"\nLoaded model for {args.year}")
            print("Model is ready for predictions")

if __name__ == "__main__":
    main()
