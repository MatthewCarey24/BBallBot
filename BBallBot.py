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

# Add the parent directory (project directory) to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cs135.hw0.hw0_split import split_into_train_and_test

def load_data(regular_season_file_path):
    df_regular = pd.read_csv(regular_season_file_path)
    return df_regular

#############################get_team_indices##################################
#
# Create dictionary of team names along with their IDs
#
# Inputs: 
#       df: CSV data from of games, scores and odds
#
###############################################################################
def get_team_indices(df):
    teams = pd.unique(df[['Away Team', 'Home Team']].values.ravel('K'))
    teams.sort()
    return {team: idx for idx, team in enumerate(teams)}

#############################create_matrices###################################
#
# Create win/loss ratio matrix for each team, to be turned into latent factors
#
# Inputs: 
#       df: CSV data from of games, scores and odds
#       team_indices: Dictionary of team names and their associated IDs
#
###############################################################################
def create_matrices(df, team_indices):
    wins_matrix = np.zeros((len(team_indices), len(team_indices)))
    games_matrix = np.zeros((len(team_indices), len(team_indices)))
    print(df.columns)
    
    for _, row in df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_pts = row['Home Score']
        away_pts = row['Away Score']
        
        home_idx = team_indices[home_team]
        away_idx = team_indices[away_team]
        
        games_matrix[home_idx, away_idx] += 1
        
        if home_pts > away_pts:
            wins_matrix[home_idx, away_idx] += 1
    
    win_ratio_matrix = np.divide(wins_matrix, games_matrix, out=np.zeros_like(wins_matrix), where=games_matrix != 0)
    return win_ratio_matrix

#############################prepare_dataset###################################
#
# Create x and y vectors consisting of each teams latent vectors
#
# Inputs: 
#       df: CSV data from of games, scores and odds
#       W: 
#       H: 
#       team_indices: Dictionary of team names and their associated IDs
#
###############################################################################
def prepare_dataset(df, W, H, team_indices):
    games = []
    labels = []
    match_info = []
    
    for _, row in df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_pts = row['Home Score']
        away_pts = row['Away Score']
        
        home_idx = team_indices[home_team]
        away_idx = team_indices[away_team]
        
        feature_vector = np.hstack([W[home_idx], H[away_idx]])
        games.append(feature_vector)
        labels.append(1 if home_pts > away_pts else 0)
        match_info.append((home_team, away_team))
    
    return np.array(games), np.array(labels), match_info

###############################testProfit######################################
#
# Calculate profit for the season given the models predictions and betting 
# stake of each game
#
# Inputs: 
#       y_pred: list of predictions made by the model
#       y_test: true game outcomes
#       stake: How much is being bet on each game
#       frac_test: Percentage of games that are being bet on
#       match_info: Information on games and odds
#
###############################################################################
def testProfit(y_pred, y_test, stake, frac_test, match_info):
    bets_won = 0
    bets_lost = 0
    start_of_test = (1-frac_test) * len(match_info)
    odds_path = 'odds_data.csv'

    df_odds = pd.read_csv(odds_path)
    profit = 0

    for i in range(len(y_test)):
        if(y_test[i] == 1):
                odds = df_odds.loc[(int(i+start_of_test)), 'Home Odds']
        elif(y_test[i] == 0):
                odds = df_odds.loc[(int(i+start_of_test)), 'Away Odds']
        if(y_pred[i] == y_test[i]):
            bets_won += 1
            # Determine which odds to use based on y_test
            # print(f'Game: {match_info[int(i+start_of_test)]}. Won odds of {odds}\n')
            if(odds > 0):
                profit += (odds / 100) * stake
            elif(odds < 0):
                profit += (100 / abs(odds)) * stake
        else:
            bets_lost += 1
            # print(f'Game: {match_info[int(i+start_of_test)]}. Lost betting against {odds}\n')
            profit -= stake
    print(f'Bets Won: {bets_won}\nBets Lost: {bets_lost}\nTotal Bets: {bets_lost+bets_won}\n')
    return profit

#######################graph_profit_vs_frac_test###############################
#
# Given a model, uses odds to determine profit over range of how much of the 
# data set was test
#
# Inputs: 
#       best_model: whichever MLP has been selected to test profits for
#       x_all_train: All feature vectors for the season (latent factors of 
#       each team in the game)
#       y_train_2d: Game outcomes, 1 for home team win, 0 for away
#       match_info: list of game information, team names and odds
#
###############################################################################
def graph_profit_vs_frac_test(best_model, X_all_train, y_train_2d, match_info):
    profits = []
    frac_tests = np.linspace(0.1, 0.9, 49)

    for frac_test in frac_tests:
        X_train, X_test = split_into_train_and_test(X_all_train, frac_test, random_state=1)
        y_train, y_test = split_into_train_and_test(y_train_2d, frac_test, random_state=1)
        best_model.fit(X_train, y_train)

        # Evaluate model
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        profit = testProfit(y_pred, y_test, 100, frac_test, match_info)
        profits.append(profit)
        print(profit)

    # Plot profit vs. frac_test
    plt.figure(figsize=(10, 6))
    plt.plot(frac_tests, profits, marker='o')
    plt.title('Profit vs. frac_test')
    plt.xlabel('frac_test')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.show()

#############################find_best_mlp#####################################
#
# Use GridSearchCV to identify best hyperparameters for the MLP classifier
#
# Inputs: 
#       x_train: All feature vectors for the season (latent factors of 
#       each team in the game)
#       y_train: Game outcomes, 1 for home team win, 0 for away
#
###############################################################################
def find_best_mlp(x_train, y_train):
        
    #         param_grid = {
#         'hidden_layer_sizes': [(50, 10), (100,), (50, 50)],
#         'activation': ['relu', 'tanh'],
#         'solver': ['adam', 'sgd'],
#         'learning_rate': ['constant', 'adaptive'],
#         'max_iter': [50, 200, 300],
#         'alpha': [0.0001, 0.001, 0.01],
#         'tol': [1e-3, 1e-4, 1e-5],
#         'batch_size': [100, 200, 500],
#         'learning_rate_init': [0.1, 0.2, 0.3],
#         'momentum': [0.0, 0.5, 0.9]
# } 
        param_grid = {'activation': ['relu'], 
                      'alpha': [0.0001], 
                      'batch_size': [100], 
                      'hidden_layer_sizes':[ (100,)],
                      'learning_rate': ['adaptive'], 
                      'learning_rate_init': [0.3], 
                      'max_iter': [200], 
                      'momentum': [0.0], 
                      'solver': ['sgd'], 
                      'tol': [0.0001]}
#         roc_grid = {
#         'hidden_layer_sizes': [(100,)],
#         'activation': ['relu'],
#         'solver': ['sgd'],
#         'learning_rate': ['constant'],
#         'max_iter': [200],
#         'alpha': [0.001],
#         'tol': [1e-3],
#         'batch_size': [200],
#         'learning_rate_init': [0.3],
#         'momentum': [0.5]
# } 


        # Train classifier
        mlp =  MLPClassifier(random_state=1)
        grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=2, scoring='accuracy', verbose=2, n_jobs=-1)
        grid_search.fit(x_train, y_train)

        print("Best Parameters:", grid_search.best_params_)
        print(f'Validation Accuracy: {grid_search.best_score_}')
        best_model = grid_search.best_estimator_
        return best_model

##############################assess_model#####################################
#
# Assess a given model using a given fraction of the data as test
#
# Inputs: 
#       best_model: Model to assess
#       x_all_train: All feature vectors for the season (latent factors of 
#       each team in the game)
#       y_train_2d: Game outcomes, 1 for home team win, 0 for away
#       frac_test: how much of the data to use as test in model assessment
#
###############################################################################
def assess_model(best_model, X_all_train, y_train_2d, frac_test):
    X_train, X_test = split_into_train_and_test(X_all_train, frac_test, random_state=1)
    y_train, y_test = split_into_train_and_test(y_train_2d, frac_test, random_state=1)
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Precision-Recall AUC
    precision_values, recall_values, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_values, precision_values)
    print(f'ROC AUC: {roc_auc}')
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)


def main():
    regular_season_file_path = 'odds_data.csv'
    df_regular = load_data(regular_season_file_path)
    num__regular_games = df_regular.shape[0]
    
    # Get team indices
    team_indices = get_team_indices(df_regular)
    
    # Create win ratio matrices
    win_ratio_matrix_regular = create_matrices(df_regular, team_indices)
    
    # Apply NMF
    nmf = NMF(n_components=5, init='random', random_state=0, max_iter=500)
    W = nmf.fit_transform(win_ratio_matrix_regular)
    H = nmf.components_.T
    
    # Prepare training dataset
    X_all_train, y_train, match_info = prepare_dataset(df_regular, W, H, team_indices)

    y_train_2d = y_train.reshape(-1,1)

    best_estimator = find_best_mlp(X_all_train, y_train_2d)
    assess_model(best_estimator, X_all_train, y_train_2d, 0.85)
    # graph_profit_vs_frac_test(best_estimator, X_all_train, y_train_2d, match_info)
    

if __name__ == '__main__':
    main()
