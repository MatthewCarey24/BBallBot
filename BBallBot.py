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

# Now import your modules
from cs135.hw0.hw0_split import split_into_train_and_test

class BetBot(): 
    def __init__(self, year):
        self.frac_test = None
        self.year = year
        self.team_indices = None
        self.match_info = None

        self.W = None
        self.H = None

        self.clf = None
        self.x_test = None
        self.y_test = None  
    #############################get_team_indices##############################
    #
    # Create dictionary of team names along with their IDs
    #
    # Inputs: 
    #       df: CSV data from of games, scores and odds
    #
    ###########################################################################
    def get_team_indices(self, df):
        teams = pd.unique(df[['Away Team', 'Home Team']].values.ravel('K'))
        teams.sort()
        self.team_indices = {team: idx for idx, team in enumerate(teams)}
    
    #############################create_matrices###############################
    #
    # Create win/loss ratio matrix for each team, to be turned into latent 
    # factors
    #
    # Inputs: 
    #       df: CSV data from of games, scores and odds
    #       team_indices: Dictionary of team names and their associated IDs
    #
    ###########################################################################
    def create_matrices(self, df, frac_test):
        num_rows = int(len(df) * (1 - frac_test))
        wins_matrix = np.zeros((len(self.team_indices), len(self.team_indices)))
        games_matrix = np.zeros((len(self.team_indices), len(self.team_indices)))
        
        for _, row in df.iloc[:num_rows].iterrows():
            home_team = row['Home Team']
            away_team = row['Away Team']
            home_pts = row['Home Score']
            away_pts = row['Away Score']
            
            home_idx = self.team_indices[home_team]
            away_idx = self.team_indices[away_team]
            
            games_matrix[home_idx, away_idx] += 1
            
            if home_pts > away_pts:
                wins_matrix[home_idx, away_idx] += 1
        
        win_ratio_matrix = np.divide(wins_matrix, games_matrix, out=np.zeros_like(wins_matrix), where=games_matrix != 0)
        return win_ratio_matrix
    
    #############################prepare_dataset###############################
    #
    # Create x and y vectors consisting of each teams latent vectors
    #
    # Inputs: 
    #       df: CSV data from of games, scores and odds
    #       W: 
    #       H: 
    #       team_indices: Dictionary of team names and their associated IDs
    #
    ###########################################################################
    def prepare_dataset(self, df):
        games = []
        labels = []
        match_info = []
        
        for _, row in df.iterrows():
            home_team = row['Home Team']
            away_team = row['Away Team']
            home_pts = row['Home Score']
            away_pts = row['Away Score']
            
            home_idx = self.team_indices[home_team]
            away_idx = self.team_indices[away_team]
            
            feature_vector = np.hstack([self.W[home_idx], self.H[away_idx]])
            games.append(feature_vector)
            labels.append(1 if home_pts > away_pts else 0)
            match_info.append((home_team, away_team))
        
        return np.array(games), np.array(labels), match_info
    
    #############################apply_nmf#####################################
    #
    # Sets the latent factor vectors
    #
    # Inputs: 
    #       win_ratio_matrix_regular: the win/loss ratio matrix
    #
    ###########################################################################
    def apply_nmf(self, win_ratio_matrix_regular):
        nmf = NMF(n_components=10, init='random', random_state=0, max_iter=500)
        self.W = nmf.fit_transform(win_ratio_matrix_regular)
        self.H = nmf.components_.T
    
    ###############################testProfit##################################
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
    ###########################################################################
    def testProfit(self, y_pred, y_test, stake, frac_test):
        bets_won = 0
        bets_lost = 0
        start_of_test = (1-frac_test) * len(self.match_info)
        odds_path = f'odds_data/odds_data_{self.year}.csv'
    
        df_odds = pd.read_csv(odds_path)
        profit = 0
    
        for i in range(len(y_test)):
            if(y_test[i] == 1):
                    odds = int(df_odds.loc[(int(i+start_of_test)), 'Home Odds'])
            elif(y_test[i] == 0):
                    odds = int(df_odds.loc[(int(i+start_of_test)), 'Away Odds'])
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
    
    #######################graph_profit_vs_frac_test###########################
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
    ###########################################################################
    def graph_profit_vs_frac_test(self, X_all, y_all_2d):
        profits = []
        frac_tests = np.linspace(0.1, 0.9, 49)
    
        for frac_test in frac_tests:
            X_train, X_test = split_into_train_and_test(X_all, frac_test, random_state=1)
            y_train, y_test = split_into_train_and_test(y_all_2d, frac_test, random_state=1)
            self.GBoost.fit(X_train, y_train)
    
            # Evaluate model
            y_pred = self.GBoost.predict(X_test)
            profit = self.testProfit(y_pred, y_test, 100, frac_test)
            profits.append(profit)
            print(profit)
    
        # Plot profit vs. frac_test
        plt.figure(figsize=(10, 6))
        plt.plot(frac_tests, profits, marker='o')
        plt.title(f'{self.year} Profit vs. frac_test')
        plt.xlabel('frac_test')
        plt.ylabel('Profit')
        plt.grid(True)
        plt.show()
    
    #############################find_best_mlp#################################
    #
    # Use GridSearchCV to identify best hyperparameters for the MLP classifier
    #
    # Inputs: 
    #       x_train: All feature vectors for the season (latent factors of 
    #       each team in the game)
    #       y_train: Game outcomes, 1 for home team win, 0 for away
    #
    ###########################################################################
    def find_best_mlp(self, x_train, y_train):
            
            param_grid = {
            'hidden_layer_sizes': [(10, 50), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [50, 200, 300],
            'alpha': [0.0001, 0.001, 0.01],
            'tol': [1e-3, 1e-4, 1e-5],
            'batch_size': [100, 200, 500],
            'learning_rate_init': [0.1, 0.2, 0.3],
            'momentum': [0.0, 0.5, 0.9]
    } 
            # param_grid = {'activation': ['relu'], 
            #               'alpha': [0.0001], 
            #               'batch_size': [200], 
            #               'hidden_layer_sizes':[ (100,)],
            #               'learning_rate': ['constant'], 
            #               'learning_rate_init': [0.3], 
            #               'max_iter': [200], 
            #               'momentum': [0.0], 
            #               'solver': ['adam'], 
            #               'tol': [0.001]}
            # 10 components
            param_grid = {'activation': ['relu'], 
                          'alpha': [0.01], 
                          'batch_size': [500], 
                          'hidden_layer_sizes':[ (100,)],
                          'learning_rate': ['adaptive'], 
                          'learning_rate_init': [0.2], 
                          'max_iter': [200], 
                          'momentum': [0.9], 
                          'solver': ['sgd'], 
                          'tol': [0.001]}
    
            # Train classifier
            mlp =  MLPClassifier(random_state=1)
            grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=2, scoring='accuracy', verbose=10, n_jobs=-1)
            grid_search.fit(x_train, y_train)
    
            print("Best Parameters:", grid_search.best_params_)
            print(f'Validation Accuracy: {grid_search.best_score_}')
            self.clf = grid_search.best_estimator_

    #############################find_best_mlp#################################
    #
    # Use GridSearchCV to identify best hyperparameters for the MLP classifier
    #
    # Inputs: 
    #       x_train: All feature vectors for the season (latent factors of 
    #       each team in the game)
    #       y_train: Game outcomes, 1 for home team win, 0 for away
    #
    ###########################################################################
    def find_best_GBoost(self, x_train, y_train):
        # param_grid = {
        #     'n_estimators': [25, 100, 200, 300],  # Number of boosting stages to be run
        #     'learning_rate': [0.01, 0.1, 0.2, 0.3],  # Learning rate shrinks the contribution of each tree
        #     'max_depth': [3, 4, 5, 6],  # Maximum depth of the individual regression estimators
        #     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        #     'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
        #     'subsample': [0.8, 0.9, 1.0],  # Fraction of samples to be used for fitting the individual base learners
        #     'max_features': [None, 'auto', 'sqrt', 'log2']  # Number of features to consider when looking for the best split
        # }
        param_grid = {
            'n_estimators': [25],
            'learning_rate': [0.01],
            'max_depth': [4],  
            'min_samples_split': [2], 
            'min_samples_leaf': [1],
            'subsample': [1.0],
            'max_features': [None]
        }
        clf = GradientBoostingClassifier(random_state=0)
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=10, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print("Best Parameters:", grid_search.best_params_)
        print(f'Validation Accuracy: {grid_search.best_score_}')
        self.clf = grid_search.best_estimator_

    
    ##############################assess_model#################################
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
    ###########################################################################
    def assess_model(self):

        y_pred = self.clf.predict(self.x_test)
        y_proba = self.clf.predict_proba(self.x_test)[:, 1]
    
        roc_auc = roc_auc_score(self.y_test, y_proba)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
    
        # Precision-Recall AUC
        precision_values, recall_values, _ = precision_recall_curve(self.y_test, y_proba)
        pr_auc = auc(recall_values, precision_values)
        print(f'ROC AUC: {roc_auc}')
        print(f'Accuracy: {accuracy}')
        print('Confusion Matrix:')
        print(conf_matrix)

    #################################train#####################################
    #
    # train the model given a data 
    #
    # Inputs: 
    #       best_model: Model to assess
    #       x_all_train: All feature vectors for the season (latent factors of 
    #       each team in the game)
    #       y_train_2d: Game outcomes, 1 for home team win, 0 for away
    #       frac_test: how much of the data to use as test in model assessment
    #
    ###########################################################################
    def train(self, df_regular, frac_test, classifier):
        self.frac_test = frac_test
        self.get_team_indices(df_regular)
        win_ratio_matrix_regular = self.create_matrices(df_regular, frac_test)
        self.apply_nmf(win_ratio_matrix_regular)

        X_all, y_all, self.match_info = self.prepare_dataset(df_regular)
        y_all_2d = y_all.reshape(-1, 1)

        x_train, self.x_test = split_into_train_and_test(X_all, frac_test, random_state=1)
        y_train, self.y_test = split_into_train_and_test(y_all_2d, frac_test, random_state=1)

        if classifier == 'GBoost':
            self.find_best_GBoost(x_train, y_train)
        elif classifier == 'mlp':
            self.find_best_mlp(x_train, y_train) 


def main():
    year = 2024
    frac_test = 0.2
    data_path = f'odds_data/odds_data_{year}.csv'
    df_2024 = pd.read_csv(data_path)

    model = BetBot(year)
    model.train(df_2024, frac_test, 'mlp')
    X_all, y_all, match_info = model.prepare_dataset(df_2024)
    y_all_2d = y_all.reshape(-1, 1)

    y_pred = model.clf.predict(model.x_test)

    model.assess_model()
    print(model.testProfit(y_pred, model.y_test, 100, frac_test))
    # model.graph_profit_vs_frac_test(X_all, y_all_2d)

if __name__ == '__main__':
    main()
