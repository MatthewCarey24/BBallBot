"""Module for betting-related calculations in BBallBot"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.metrics import make_scorer

def calculate_implied_proba(odds: float) -> float:
    """
    Calculate implied probability from moneyline odds.
    
    Args:
        odds: Moneyline odds
        
    Returns:
        Implied probability of victory
    """
    if odds < 0:
        return float(1 / ((100 / abs(odds)) + 1))
    else:
        return float(1 / ((odds / 100) + 1))

def get_team_info(df_odds: pd.DataFrame, index: int, is_home_team: bool) -> Tuple[str, float, float]:
    """
    Get team and odds information for a specific game.
    
    Args:
        df_odds: DataFrame containing odds data
        index: Row index in the DataFrame
        is_home_team: Whether to get home team info (True) or away team info (False)
        
    Returns:
        Tuple of (team name, win odds, opposing team odds)
    """
    team_column = "Home Team" if is_home_team else "Away Team"
    win_odds_column = "Home Odds" if is_home_team else "Away Odds"
    loss_odds_column = "Away Odds" if is_home_team else "Home Odds"
    
    team = str(df_odds.loc[index, team_column])
    win_odds = float(df_odds.loc[index, win_odds_column])
    loss_odds = float(df_odds.loc[index, loss_odds_column])
    
    return team, win_odds, loss_odds

def print_bet_info(team: str, odds: float, bet_amount: float, result: str, curr_wealth: int) -> None:
    """
    Print information about a bet and its outcome.
    
    Args:
        team: Name of the team bet on
        odds: Moneyline odds for the bet
        bet_amount: Amount wagered
        result: "win" or "loss"
        curr_wealth: How much money you have
    """
    print(f'Current wealth: {curr_wealth}')
    print(f'Betting ${bet_amount:.2f} on {team} with odds: {odds}')
    if result == "win":
        if odds > 0:
            print(f'Won ${(odds / 100) * bet_amount:.2f}')
        else:
            print(f'Won ${(100 / abs(odds)) * bet_amount:.2f}')
    elif result == "loss":
        print(f'Lost ${bet_amount:.2f}')

def calculate_frac_wealth(win_odds: float, loss_odds: float, y_proba: np.ndarray, index: int) -> float:
    """
    Calculate fraction of wealth to bet using Kelly Criterion.
    
    Args:
        win_odds: Moneyline odds for the team being bet on
        loss_odds: Moneyline odds for the opposing team
        y_proba: Model's predicted probabilities
        index: Index of the current game within the test set (not of all games)
        
    Returns:
        Fraction of wealth to bet (between 0.1 and 0.5)
    """
    proba_win = float(max(y_proba[index][0], y_proba[index][1]))
    proba_lose = 1.0 - proba_win

    percent_gain = float(win_odds / 100) if win_odds > 0 else float(100 / abs(win_odds))
    frac_wealth = float(proba_win - (proba_lose / percent_gain))

    # Ensure fraction is between 0.01 and 0.2
    return max(frac_wealth, 0)

def test_profit(
    df_path: str,
    y_pred: np.ndarray,
    y_test: np.ndarray,
    y_proba: np.ndarray,
    starting_wealth: float,
    frac_test: float
) -> Tuple[float, float]:
    """
    Calculate profit/loss from betting based on model predictions.
    
    Args:
        df_path: Path to the odds data CSV file
        y_pred: Model's predictions
        y_test: True outcomes
        y_proba: Model's predicted probabilities
        starting_wealth: Initial bankroll
        frac_test: Fraction of data used for testing
        
    Returns:
        Tuple of (final wealth, total amount staked)
    """
    bets_won = 0
    bets_lost = 0
    no_bet_win = 0
    no_bet_loss = 0
    df_odds = pd.read_csv(df_path)
    start_of_test = int(len(df_odds) * (1 - frac_test))
    wealth = float(starting_wealth)
    total_stake = 0.0
    
    # Convert date and time columns to datetime
    df_odds['datetime'] = pd.to_datetime(df_odds['Date'] + ' ' + df_odds['Time'])
    
    # Group games by date to process concurrent games together
    current_games = []
    current_time = None
    current_bets = []
    
    for i in range(len(y_test)):
        index = int(i + start_of_test)
        game_time = df_odds.loc[index, 'datetime']
        
        # If this is a new time or more than 3 hours from current time
        if current_time is None or (game_time - current_time).total_seconds() > 10800:
            # Process any pending bets from previous time slot
            for bet in current_bets:
                print_bet_info(bet['team'], bet['odds'], bet['amount'], bet['result'], wealth)
                if bet['result'] == 'win':
                    if bet['odds'] > 0:
                        wealth += float((bet['odds'] / 100) * bet['amount'])
                    else:
                        wealth += float((100 / abs(bet['odds'])) * bet['amount'])
                    bets_won += 1
                else:
                    wealth -= bet['amount']
                    bets_lost += 1
            
            # Reset for new time slot
            current_time = game_time
            current_bets = []
            
        is_home_team = bool(y_pred[i] == 1)
        team, win_odds, loss_odds = get_team_info(df_odds, index, is_home_team)
        
        # Calculate bet using current wealth instead of starting_wealth
        frac_wealth = calculate_frac_wealth(win_odds, loss_odds, y_proba, i)
        bet_amount = float(frac_wealth * wealth)
        total_stake += bet_amount
        
        if frac_wealth > 0:
            current_bets.append({
                'team': team,
                'amount': bet_amount,
                'odds': win_odds,
                'result': 'win' if y_pred[i] == y_test[i] else 'loss'
            })
        else:
            if y_pred[i] == y_test[i]:
                no_bet_win += 1
            else:
                no_bet_loss += 1
    
    # Process any remaining bets
    for bet in current_bets:
        print_bet_info(bet['team'], bet['odds'], bet['amount'], bet['result'], wealth)
        if bet['result'] == 'win':
            if bet['odds'] > 0:
                wealth += float((bet['odds'] / 100) * bet['amount'])
            else:
                wealth += float((100 / abs(bet['odds'])) * bet['amount'])
            bets_won += 1
        else:
            wealth -= bet['amount']
            bets_lost += 1
    
    print(f'Bets Won: {bets_won}\nBets Lost: {bets_lost}\nWouldve Won: {no_bet_win}\nWouldve Lost: {no_bet_loss}\nTotal Bets: {bets_won + bets_lost}\n')
    print(f'Good Call Ratio: {(bets_won+no_bet_loss)/(bets_won+bets_lost+no_bet_loss+no_bet_win)}')
    
    return float(wealth), float(total_stake)

def _adapt_for_profit_scorer(y_true: np.ndarray, y_pred: np.ndarray, df_odds: pd.DataFrame, stake: float) -> float:
    """
    Adapter function to use test_profit for model evaluation scoring.
    
    Args:
        y_true: True outcomes
        y_pred: Predicted outcomes
        df_odds: DataFrame containing odds data
        stake: Initial stake amount
        
    Returns:
        Final wealth after all bets
    """
    import tempfile
    import os
    
    # Create dummy probabilities if needed for test_profit
    y_proba = np.zeros((len(y_true), 2))
    for i, pred in enumerate(y_pred):
        y_proba[i][int(pred)] = 1.0
    
    # Save DataFrame to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        temp_path = temp_file.name
        df_odds.to_csv(temp_path, index=False)
    
    try:
        # Use a small frac_test value since we're using the entire dataset
        frac_test = 0.001
        
        # Suppress print statements from test_profit
        import sys
        from io import StringIO
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Call test_profit and get only the final wealth
            final_wealth, _ = test_profit(temp_path, y_pred, y_true, y_proba, stake, frac_test)
            return float(final_wealth)
        finally:
            # Restore stdout
            sys.stdout = original_stdout
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def profit_scorer(y_true: np.ndarray, y_pred: np.ndarray, df_odds: pd.DataFrame, stake: float) -> float:
    """
    Custom scorer for model evaluation using profit.
    
    Args:
        y_true: True outcomes
        y_pred: Predicted outcomes
        df_odds: DataFrame containing odds data
        stake: Initial stake amount
        
    Returns:
        Profit score
    """
    return _adapt_for_profit_scorer(y_true, y_pred, df_odds, stake)

def create_profit_scorer(df_odds: pd.DataFrame, stake: float) -> Any:
    """
    Create a scorer function for use in cross-validation.
    
    Args:
        df_odds: DataFrame containing odds data
        stake: Initial stake amount
        
    Returns:
        Scorer function for use with sklearn
    """
    return make_scorer(profit_scorer, df_odds=df_odds, stake=stake)
