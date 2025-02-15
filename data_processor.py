"""Module for data processing functions in BBallBot"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
from sklearn.decomposition import NMF
from config import NMF_MAX_ITER, NMF_INIT, RANDOM_STATE

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

def calculate_team_stats(df: pd.DataFrame, team: str, up_to_index: Any, window: int = 10) -> Dict[str, float]:
    """
    Calculate rolling statistics for a team.
    
    Args:
        df: DataFrame containing game data
        team: Team name
        up_to_index: Index up to which to calculate stats
        window: Number of recent games to consider
        
    Returns:
        Dictionary of team statistics
    """
    idx = int(up_to_index) if not isinstance(up_to_index, int) else up_to_index
    recent_games = df.iloc[:idx]
    home_games = recent_games[recent_games['Home Team'] == team].tail(window)
    away_games = recent_games[recent_games['Away Team'] == team].tail(window)
    
    total_games = pd.concat([
        home_games[['Home Score', 'Away Score']].rename(columns={'Home Score': 'Team Score', 'Away Score': 'Opp Score'}),
        away_games[['Away Score', 'Home Score']].rename(columns={'Away Score': 'Team Score', 'Home Score': 'Opp Score'})
    ])
    
    if len(total_games) == 0:
        return {
            'avg_score': 0.0,
            'avg_allowed': 0.0,
            'win_pct': 0.0,
            'score_diff': 0.0
        }
    
    stats = {
        'avg_score': float(total_games['Team Score'].mean()),
        'avg_allowed': float(total_games['Opp Score'].mean()),
        'win_pct': float((total_games['Team Score'] > total_games['Opp Score']).mean()),
        'score_diff': float((total_games['Team Score'] - total_games['Opp Score']).mean())
    }
    
    return stats

def get_h2h_stats(df: pd.DataFrame, home_team: str, away_team: str, up_to_index: Any, window: int = 5) -> Dict[str, float]:
    """
    Calculate head-to-head statistics between two teams.
    
    Args:
        df: DataFrame containing game data
        home_team: Home team name
        away_team: Away team name
        up_to_index: Index up to which to calculate stats
        window: Number of recent matchups to consider
        
    Returns:
        Dictionary of head-to-head statistics
    """
    idx = int(up_to_index) if not isinstance(up_to_index, int) else up_to_index
    recent_games = df.iloc[:idx]
    
    h2h_games = recent_games[
        ((recent_games['Home Team'] == home_team) & (recent_games['Away Team'] == away_team)) |
        ((recent_games['Home Team'] == away_team) & (recent_games['Away Team'] == home_team))
    ].tail(window)
    
    if len(h2h_games) == 0:
        return {
            'home_win_pct': 0.5,
            'avg_point_diff': 0.0
        }
    
    home_wins = []
    point_diffs = []
    
    for _, game in h2h_games.iterrows():
        if game['Home Team'] == home_team:
            home_wins.append(1 if game['Home Score'] > game['Away Score'] else 0)
            point_diffs.append(game['Home Score'] - game['Away Score'])
        else:
            home_wins.append(1 if game['Away Score'] > game['Home Score'] else 0)
            point_diffs.append(game['Away Score'] - game['Home Score'])
    
    return {
        'home_win_pct': float(np.mean(home_wins)),
        'avg_point_diff': float(np.mean(point_diffs))
    }

def prepare_x_y(team_indices: Dict[str, int], df: pd.DataFrame, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature vectors and labels for model training.
    
    Args:
        team_indices: Dictionary mapping team names to their indices
        df: DataFrame containing game matchups and outcomes
        W: Home team latent vectors from NMF
        H: Away team latent vectors from NMF
        
    Returns:
        Tuple of (X, y) where X contains feature vectors and y contains labels
    """
    games: List[np.ndarray] = []
    labels: List[int] = []
    
    for idx, row in df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_idx = team_indices[home_team]
        away_idx = team_indices[away_team]
        
        # Get team statistics
        home_stats = calculate_team_stats(df, home_team, idx)
        away_stats = calculate_team_stats(df, away_team, idx)
        h2h_stats = get_h2h_stats(df, home_team, away_team, idx)
        
        # Create feature vector
        home_features = [
            home_stats['avg_score'],
            home_stats['avg_allowed'],
            home_stats['win_pct'],
            home_stats['score_diff'],
            row['Home Odds']
        ]
        
        away_features = [
            away_stats['avg_score'],
            away_stats['avg_allowed'],
            away_stats['win_pct'],
            away_stats['score_diff'],
            row['Away Odds']
        ]
        
        h2h_features = [
            h2h_stats['home_win_pct'],
            h2h_stats['avg_point_diff']
        ]
        
        # Combine all features
        home_vector = np.append(W[home_idx], home_features)
        away_vector = np.append(H[away_idx], away_features)
        feature_vector = np.hstack([home_vector, away_vector, h2h_features])
        
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
