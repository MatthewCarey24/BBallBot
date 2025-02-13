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

def print_bet_info(team: str, odds: float, bet_amount: float, result: str) -> None:
    """
    Print information about a bet and its outcome.
    
    Args:
        team: Name of the team bet on
        odds: Moneyline odds for the bet
        bet_amount: Amount wagered
        result: "win" or "loss"
    """
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
        index: Index of the current game
        
    Returns:
        Fraction of wealth to bet (between 0.1 and 1.0)
    """
    proba_win = float(max(y_proba[index][0], y_proba[index][1]))
    proba_lose = 1.0 - proba_win

    percent_gain = float(win_odds / 100) if win_odds > 0 else float(100 / abs(win_odds))
    frac_wealth = float(proba_win - (proba_lose / percent_gain))

    # Ensure fraction is between 0.1 and 1.0
    return max(min(frac_wealth, 1.0), 0.1)

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
    
    for i in range(len(y_test)):
        index = int(i + start_of_test)
        is_home_team = bool(y_pred[i] == 1)
        team, win_odds, loss_odds = get_team_info(df_odds, index, is_home_team)
        
        frac_wealth = calculate_frac_wealth(win_odds, loss_odds, y_proba, i)
        bet_amount = float(frac_wealth * wealth)
        total_stake += bet_amount
        
        if y_pred[i] == y_test[i] and frac_wealth > 0:
            if win_odds > 0:
                wealth += float((win_odds / 100) * bet_amount)
            else:
                wealth += float((100 / abs(win_odds)) * bet_amount)
            bets_won += 1
        elif frac_wealth > 0:
            wealth -= bet_amount
            bets_lost += 1
        
        if frac_wealth == 0:
            if y_pred[i] == y_test[i]:
                no_bet_win += 1
            else:
                no_bet_loss += 1
    
    print(f'Bets Won: {bets_won}\nBets Lost: {bets_lost}\nWouldve Won: {no_bet_win}\nWouldve Lost: {no_bet_loss}\nTotal Bets: {bets_won + bets_lost}\n')
    print(f'Good Call Ratio: {(bets_won+no_bet_loss)/(bets_won+bets_lost+no_bet_loss+no_bet_win)}')
    
    return float(wealth), float(total_stake)

def calculate_profit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df_odds: pd.DataFrame,
    starting_wealth: float
) -> float:
    """
    Calculate profit for model evaluation.
    
    Args:
        y_true: True outcomes
        y_pred: Predicted outcomes
        df_odds: DataFrame containing odds data
        starting_wealth: Initial bankroll
        
    Returns:
        Final wealth after all bets
    """
    wealth = float(starting_wealth)
    for i in range(len(y_true)):
        odds_val = df_odds.loc[i, 'Home Odds'] if y_true[i] == 1 else df_odds.loc[i, 'Away Odds']
        odds = float(odds_val)
        
        if odds > 0:
            frac_wealth = float(0.5 - (0.5 / (odds/100.0)))
        else:
            abs_odds = abs(float(odds))
            frac_wealth = float(0.5 - (0.5 / (100.0/abs_odds)))
        
        bet_amount = float(wealth * frac_wealth)
        
        if y_pred[i] == y_true[i]:
            if odds > 0:
                wealth += float((odds / 100.0) * bet_amount)
            else:
                wealth += float((100.0 / abs(float(odds))) * bet_amount)
        else:
            wealth -= bet_amount
    
    return float(wealth)

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
    return calculate_profit(y_true, y_pred, df_odds, stake)

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
