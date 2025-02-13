# BBallBot

A machine learning system for basketball betting predictions using historical odds data.

## Project Structure

- `main.py` - Main entry point for training and using models
- `config.py` - Configuration constants and file paths
- `data_processor.py` - Data loading and preprocessing functions
- `model_trainer.py` - Model training and optimization using Optuna
- `betting.py` - Betting logic and profit calculations
- `utils.py` - Utility functions for data manipulation
- `odds_data/` - Directory containing historical odds data CSV files

## Components

### Data Processing
- Team indexing and win/loss matrix creation
- Feature preparation using NMF (Non-negative Matrix Factorization)
- Data splitting for training and testing

### Model Training
- Neural network model using scikit-learn's MLPClassifier
- Hyperparameter optimization with Optuna
- Model evaluation using accuracy and profit metrics

### Betting Logic
- Kelly Criterion for bet sizing
- Profit calculation and tracking
- Implied probability calculations from odds

## Usage

### Training a New Model

```bash
python main.py --year 2022 --train
```

This will:
1. Load odds data for the specified year
2. Train a model using Optuna for hyperparameter optimization
3. Save the best model and its parameters
4. Print accuracy and profit metrics

### Loading a Trained Model

```bash
python main.py --year 2022
```

This will load a previously trained model for the specified year.

## Requirements

- Python 3.8+
- NumPy
- Pandas
- scikit-learn
- Optuna
- joblib

## Data Format

The odds data CSV files should have the following columns:
- Home Team
- Away Team
- Home Score
- Away Score
- Home Odds
- Away Odds
