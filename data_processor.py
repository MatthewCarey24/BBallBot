"""Module for data processing functions in BBallBot"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from betting import calculate_implied_proba
from config import NMF_INIT, NMF_MAX_ITER, RANDOM_STATE


def get_team_indices(df: pd.DataFrame) -> Dict[str, int]:
    """
    Create a mapping of team names to indices.

    Args:
        df: DataFrame containing 'Away Team' and 'Home Team' columns

    Returns:
        Dictionary mapping team names to their indices
    """
    teams = pd.unique(df[['Away Team', 'Home Team']].values.ravel('K'))
    teams.sort()
    return {team: idx for idx, team in enumerate(teams)}

def create_win_loss_matrix(team_indices: Dict[str, int], df: pd.DataFrame, frac_test: float) -> np.ndarray:
    """
    Create a win/loss matrix where index (i,j) is the ratio of games that team i won against team j at home.

    Args:
        team_indices: Dictionary mapping team names to their indices
        df: DataFrame containing game matchups and outcomes
        frac_test: Fraction of the season to use as test set

    Returns:
        Win/loss ratio matrix
    """
    # Sort DataFrame by date to ensure consistent ordering
    df = df.sort_index()
    num_rows = int(len(df) * (1 - frac_test))
    num_teams = len(team_indices)
    wins_matrix = np.zeros((num_teams, num_teams))
    games_matrix = np.zeros((num_teams, num_teams))

    for _, row in df.iloc[:num_rows].iterrows():
        home_idx = team_indices[row['Home Team']]
        away_idx = team_indices[row['Away Team']]
        home_pts, away_pts = row['Home Score'], row['Away Score']
        games_matrix[home_idx, away_idx] += 1
        if home_pts > away_pts:
            wins_matrix[home_idx, away_idx] += 1

    return np.divide(wins_matrix, games_matrix, out=np.zeros_like(wins_matrix), where=games_matrix != 0)

def prepare_x_y(team_indices: Dict[str, int], df: pd.DataFrame, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature vectors and labels for model training.

    Odds are expressed as implied probabilities in [0, 1] rather than raw
    American-odds integers.  This removes the discontinuity at ±100 and
    produces features on a uniform [0, 1] scale that the MLP can learn from
    more easily.

    Args:
        team_indices: Dictionary mapping team names to their indices
        df: DataFrame containing game matchups and outcomes
        W: Home team latent vectors from NMF (shape: n_teams × n_components)
        H: Away team latent vectors from NMF (shape: n_teams × n_components)

    Returns:
        Tuple of (X, y) where:
            X: feature matrix of shape (n_games, 2*(n_components+1)) —
               last two features are home and away implied win probabilities.
            y: binary label array (1 = home win).
    """
    games: List[np.ndarray] = []
    labels: List[int] = []

    for _, row in df.iterrows():
        home_idx = team_indices[row['Home Team']]
        away_idx = team_indices[row['Away Team']]
        # Handle missing or invalid odds values.
        # '-' means no line was posted; we use 0.5 (even money) as a neutral
        # prior rather than 0, which would break the implied-probability scale.
        home_odds_raw = row['Home Odds']
        away_odds_raw = row['Away Odds']

        home_odds = 100 if home_odds_raw == '-' else float(home_odds_raw)
        away_odds = 100 if away_odds_raw == '-' else float(away_odds_raw)

        home_implied = calculate_implied_proba(home_odds)
        away_implied = calculate_implied_proba(away_odds)

        home_vector = np.append(W[home_idx], home_implied)
        away_vector = np.append(H[away_idx], away_implied)

        feature_vector = np.hstack([home_vector, away_vector])
        games.append(feature_vector)
        labels.append(1 if row['Home Score'] > row['Away Score'] else 0)

    return np.array(games), np.array(labels)

def create_features(df: pd.DataFrame, frac_test: float, n_components: int, alpha_H: float, alpha_W: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create feature matrices using NMF decomposition of the win/loss matrix.

    Args:
        df: DataFrame containing game data
        frac_test: Fraction of data to use for testing
        n_components: Number of components for NMF
        alpha_H: L2 regularization parameter for H matrix
        alpha_W: L2 regularization parameter for W matrix

    Returns:
        Tuple of (X, y) arrays for model training
    """
    team_indices = get_team_indices(df)
    win_ratio_matrix = create_win_loss_matrix(team_indices, df, frac_test)

    nmf = NMF(
        n_components=n_components,
        init=NMF_INIT,
        alpha_H=alpha_H,
        alpha_W=alpha_W,
        random_state=RANDOM_STATE,
        max_iter=NMF_MAX_ITER
    )

    W = nmf.fit_transform(win_ratio_matrix)
    H = nmf.components_.T

    return prepare_x_y(team_indices, df, W, H)
