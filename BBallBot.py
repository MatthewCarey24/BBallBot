import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import optuna
import sys
import os
import joblib
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cs135.hw0.hw0_split import split_into_train_and_test

def get_team_indices(df):
    teams = pd.unique(df[['Away Team', 'Home Team']].values.ravel('K'))
    teams.sort()
    return {team: idx for idx, team in enumerate(teams)}

############################create_win_loss_matrix#############################
#
# Use the df, team indices dict and frac test to create a win/loss matrix to
# train on
#
# Inputs: 
#           team_indices: dict of team name and their associated index
#           df: data frame that contains all game matchups and outcomes
#           frac_test: fraction of the season that is being used as test
# 
# Outputs: 
#           win_loss_matrix: num_teams x num_teams diagonal matrix, where index
#           (i,j) is the ratio of games that team i won against team j at home
#
###############################################################################
def create_win_loss_matrix(team_indices, df, frac_test):
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

#################################prepare_x_y###################################
#
# Use the df, team indices dict and latent vectors of each team to create the X
# vector,  where each item is (home latent vector, away latent vector), as well
# as create the y vector, which is 1 if the home team won, 0 else
#
# Inputs: 
#           team_indices: dict of team name and their associated index
#           df: data frame that contains all game matchups and outcomes
#           W: Home team latent vectors, derived from win/loss matrix
#           H: Away team latent vectors, derived from win/loss matrix
# 
# Outputs: 
#           X: total_games by 2 array where the ith element is (home latent 
#           vector, away latent vector) for the ith game
#           Y: total_games by 1 array of 1 (home team won) or 0 (away team won)
#
###############################################################################

def prepare_x_y(team_indices, df, W, H):
    games, labels = [], []
    for _, row in df.iterrows():
        home_idx, away_idx = team_indices[row['Home Team']], team_indices[row['Away Team']]
        # home_odds = row['Home Odds']
        # away_odds = row['Away Odds']

        # home_vector = np.append(W[home_idx], home_odds)
        # away_vector = np.append(H[away_idx], away_odds)

        home_vector = W[home_idx]
        away_vector = H[away_idx]

        feature_vector = np.hstack([home_vector, away_vector])
        games.append(feature_vector)
        labels.append(1 if row['Home Score'] > row['Away Score'] else 0)
    return np.array(games), np.array(labels)


#################################test_profit###################################
#
# Use the df, y predictions and true values, betting stake and frac test to 
# determine the profit or loss if the model was used for this season
#
# Inputs: 
#           df: data frame that contains all game matchups and outcomes
#           y_pred: model predictions of test values
#           y_test: true test game outcomes
#           stake: how much money is being bet each game
#           frac_test: how much of the full data frame y_test/y_pred are
# 
# Outputs: 
#           profit: total gain or loss if you had followed the models 
#           predictions for the last {frac_test} of the season
#
###############################################################################
def test_profit(df_path, y_pred, y_test, stake, frac_test):
    bets_won = 0
    bets_lost = 0
    df_odds = pd.read_csv(df_path)
    start_of_test = (1-frac_test) * df_odds.shape[0]
    profit = 0

    for i in range(len(y_test)):
        if(y_test[i] == 1):
                odds = df_odds.loc[(int(i+start_of_test)), 'Home Odds']
        elif(y_test[i] == 0):
                odds = df_odds.loc[(int(i+start_of_test)), 'Away Odds']
        if(y_pred[i] == y_test[i]):
            bets_won += 1
            # Determine which odds to use based on y_test
            if(odds > 0):
                profit += (odds / 100) * stake
            elif(odds < 0):
                profit += (100 / abs(odds)) * stake
        else:
            bets_lost += 1
            profit -= stake
    print(f'Bets Won: {bets_won}\nBets Lost: {bets_lost}\nTotal Bets: {bets_lost+bets_won}\n')
    return profit

import matplotlib.pyplot as plt

#############################plot_learning_curve###############################
#
# Plots the SGD of the MLP fitting to see if its over or understepping
#
# Inputs: 
#           history: the training loss over iterations
# 
# Outputs: 
#           Displays a graph of the loss after n iterations
#
###############################################################################
def plot_learning_curve(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

###################################objective###################################
#
# Objective function to be optimized by the optuna study. Suggests 
# hyperparameters of the NMF feature formation and MLP classifier, creates a 
# pipeline, and uses CV to maximise accuracy of the model
#
# Inputs: 
#           df: data frame that contains all game matchups and outcomes
#           frac_test: how much of the full data frame y_test/y_pred are
#           trial: the trial of the optuna study
# 
# Outputs: 
#           The cross validation accuracy score of the model with given 
#           hyperparameters
#
###############################################################################
def objective(df_path, frac_test, trial):
    df = pd.read_csv(df_path)
    # Parameters
    nmf_n_components = trial.suggest_int('nmf_n_components', 5, 10)
    nmf_alpha_H=trial.suggest_float('alpha_H', 0.0001, 0.1001, step=0.005)
    nmf_alpha_W=trial.suggest_float('alpha_W', 0.0001, 0.1001, step=0.005)
    mlp_params = {
        'first_layer_neurons': trial.suggest_int('first_layer_neurons', 10, 100, step=10),
        'second_layer_neurons': trial.suggest_int('second_layer_neurons', 10, 100, step=10),
        # 'third_layer_neurons': trial.suggest_int('third_layer_neurons', 10, 100, step=10),
        'activation': trial.suggest_categorical('activation', ['tanh', 'relu']),
        'solver': trial.suggest_categorical('solver', ['sgd', 'adam']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
        'max_iter':trial.suggest_int('max_iter', 2000, 2001),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.1001, step=0.005),
    }

    nmf = NMF(n_components=nmf_n_components, init='random', alpha_H=nmf_alpha_H, alpha_W=nmf_alpha_W, random_state=0, max_iter=1000)
    win_ratio_matrix = create_win_loss_matrix(get_team_indices(df), df, frac_test)
    W, H = nmf.fit_transform(win_ratio_matrix), nmf.components_.T

    X, y = prepare_x_y(get_team_indices(df), df, W, H)
    y_2d = y.reshape(-1,1)

    # Save x and y
    trial_id = trial.number
    np.save(f'x_trial_{trial_id}.npy', X)
    np.save(f'y_trial_{trial_id}.npy', y)

    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(mlp_params['first_layer_neurons'], mlp_params['second_layer_neurons']), 
                              activation=mlp_params['activation'],
                              solver=mlp_params['solver'],
                              alpha=mlp_params['alpha'],
                              learning_rate=mlp_params['learning_rate'],
                              learning_rate_init=mlp_params['learning_rate_init'],
                              max_iter=mlp_params['max_iter'],
                              random_state=0))
    ])

    # Fit model
    X_train, X_test = split_into_train_and_test(X, frac_test, random_state=1)
    y_train, y_test = split_into_train_and_test(y_2d, frac_test, random_state=1)
    pipeline.fit(X_train, y_train.ravel())
    y_pred = pipeline.predict(X_test)

    # Track learning curve if it goes over 1000 iter to see whats going on 
    history = {
        'train_loss': pipeline.named_steps['clf'].loss_curve_ # or similar method
    }
    if(len(history['train_loss']) >= 1000):
        plot_learning_curve(history)

    return test_profit(df_path, y_pred, y_test, 100, frac_test)
    # return cross_val_score(pipeline, X, y, cv=5, scoring='accuracy').mean()


##############################save_best_trial##################################
#
# Use the best trial from the optuna study to save the model and parameters to
# a pkl file so they can be accessed without runnng the long study again
#
# Inputs: 
#           best_trial: the best trial found from the optuna study
#           year: the year being used for the file name
# 
# Outputs: 
#           Nothing returned. Outputs the trial and parameters to pkl files
#
###############################################################################
def save_best_trial(best_trial, year):
    mlp_params = best_trial.params

    best_classifier = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(mlp_params['first_layer_neurons'], mlp_params['second_layer_neurons']), 
                              activation=mlp_params['activation'],
                              solver=mlp_params['solver'],
                              alpha=mlp_params['alpha'],
                              learning_rate=mlp_params['learning_rate'],
                              learning_rate_init=mlp_params['learning_rate_init'],
                              max_iter=mlp_params['max_iter'],
                              random_state=0))
    ])

    # Save the best model
    model_filename = f'best_mlp_model_{year}.pkl'
    joblib.dump(best_classifier, model_filename)
    print(f"Model saved to {model_filename}")

    # Save the best trial parameters
    trial_filename = f'best_trial_{year}.pkl'
    with open(trial_filename, 'wb') as f:
        pickle.dump(best_trial, f)
    print(f"Best trial parameters saved to {trial_filename}")

    # Save the best trial parameters
    params_filename = f'best_trial_params_{year}.pkl'
    with open(params_filename, 'wb') as f:
        pickle.dump(best_trial.params, f)
    print(f"Best trial parameters saved to {params_filename}")



def main():
    frac_test = 0.2
    year = 2024
    df_path = f'odds_data/odds_data_{year}.csv'
    df = pd.read_csv(df_path)

    def objective_function(trial):
        return objective(df_path, frac_test=frac_test, trial=trial)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_function, n_trials=10)

    best_trial = study.best_trial
    print(f"Best trial: Value={best_trial.value}, Params={best_trial.params}")
    save_best_trial(best_trial, year)


     # Load the model
    best_classifier = joblib.load(f'best_mlp_model_{year}.pkl')
    with open(f'best_trial_params_{year}.pkl', 'rb') as f:
        best_params = pickle.load(f)

    with open(f'best_trial_{year}.pkl', 'rb') as f:
        best_trial = pickle.load(f)
    
    # Load x and y from the best trial
    trial_id = best_trial.number
    X = np.load(f'x_trial_{trial_id}.npy')
    y = np.load(f'y_trial_{trial_id}.npy')
    y_2d = y.reshape(-1,1)

    X_train, X_test = split_into_train_and_test(X, frac_test, random_state=1)
    y_train, y_test = split_into_train_and_test(y_2d, frac_test, random_state=1)

    best_classifier.fit(X_train, y_train.ravel())
    y_pred = best_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    profit = test_profit(df_path, y_pred, y_test, 100, frac_test)

    print(f'frac_test={frac_test}\nAccuracy={accuracy}\nProfit={profit}')

if __name__ == "__main__":
    main()
