import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc
)
import sys
import os
import matplotlib.pyplot as plt

from load_other_team_features import load_team_stats
from BBallBot import create_matrices, get_team_indices, load_data

# Add the parent directory (project directory) to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cs135.projB.src_starter.AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from cs135.projB.src_starter.CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem

class BetBot(CollabFilterOneVectorPerItem):
    def __init__(self, home_team_features, away_team_features, **kwargs):
        super().__init__(**kwargs)
        self.home_team_stats = home_team_features
        self.away_team_stats = away_team_features

    def prepare_features(self, home_ids, away_ids):
        latent_vectors_predictions = super().predict(home_ids, away_ids)  # Using CollabFilterOneVectorperItem's predict
        latent_vectors_predictions = latent_vectors_predictions.reshape(-1, 1) 

        home_team_stats = self.home_team_stats.loc[home_ids].values
        away_team_stats = self.away_team_stats.loc[away_ids].values


        combined_features = np.concatenate([latent_vectors_predictions, home_team_stats, away_team_stats], axis=1)
        return combined_features
        


if __name__ == '__main__':
    n_teams = 30
    n_home_teams = n_away_teams = n_teams

    year = 2024

    teams_outcomes_odds_df = load_data(f'odds_data/odds_data_{year}.csv')

    team_indices = get_team_indices(teams_outcomes_odds_df)

    home_team_features, away_team_features = load_team_stats()

    win_loss_matrix = create_matrices(teams_outcomes_odds_df, team_indices)
    train_tuple = (
        np.array([[i for i in range(n_home_teams)]]),
        np.array([[j for j in range(n_away_teams)]]),
        np.array([win_loss_matrix[i, j] for i in range(n_home_teams) for j in range(n_away_teams)])
    )    
    train_labels = train_tuple[2] 

    K = 5

    # Instantiate the recommender system
    recommender = BetBot(home_team_features, away_team_features, step_size=0.1, n_epochs=10, batch_size=100, n_factors=K, alpha=0.01)
    recommender.init_parameter_dict(n_home_teams, n_away_teams, train_tuple)

    recommender.fit(train_data_tuple=train_tuple)

    # Prepare features
    train_features = recommender.prepare_features(train_tuple[0], train_tuple[1])

    # Train the classifier
    recommender.train_classifier(train_features, train_labels)
