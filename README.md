# BBallBot

BBallBot is a machine learning-based NBA game prediction and betting system that combines web scraping, data processing, and neural networks to make informed betting decisions.

## Components

### Web Scraping (`scrape_odds.py` & `enhanced_scrape.py`)

The scraping component collects NBA game data from oddsportal.com, including:
- Team matchups
- Game scores
- Betting odds (home and away)

Features:
- Uses Selenium WebDriver for dynamic content loading
- Implements smart scrolling to ensure all content is loaded
- Saves data to CSV files organized by season
- Handles pagination and data extraction robustly
- Includes error handling and progress tracking

### Prediction System (`BBallBot.py`)

The core prediction system uses machine learning to analyze historical game data and make predictions:

#### Feature Engineering
- Creates team performance matrices using Non-negative Matrix Factorization (NMF)
- Generates latent vectors representing team characteristics
- Combines team vectors with betting odds for feature creation

#### Machine Learning Model
- Uses Multi-layer Perceptron (MLP) classifier
- Implements hyperparameter optimization using Optuna
- Includes model calibration for probability estimates
- Cross-validates performance using accuracy metrics

#### Betting Strategy
- Implements Kelly Criterion for optimal bet sizing
- Calculates implied probabilities from betting odds
- Tracks betting performance and profitability
- Provides detailed statistics on betting outcomes

## Usage

1. Scrape historical NBA game data:
```python
python enhanced_scrape.py
```

2. Train and evaluate the prediction model:
```python
python BBallBot.py
```

## Data Structure

The system stores scraped data in the `odds_data` directory with files named by season (e.g., `odds_data_2021.csv`). Each file contains:
- Home Team
- Away Team
- Home Score
- Away Score
- Home Odds
- Away Odds

## Model Performance Metrics

The system tracks several key performance indicators:
- Prediction accuracy
- Betting profitability
- ROI on placed bets
- Win/loss ratio
- Kelly Criterion efficiency

## Requirements

- Python 3.x
- Selenium WebDriver
- Chrome WebDriver
- BeautifulSoup4
- pandas
- numpy
- scikit-learn
- optuna
- matplotlib

## Note

This project is for educational purposes only. Please be aware of and comply with all local laws and regulations regarding sports betting.
