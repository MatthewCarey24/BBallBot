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

def load_data(regular_season_file_path, postseason_file_path):
    df_regular = pd.read_csv(regular_season_file_path)
    df_postseason = pd.read_csv(postseason_file_path)
    return df_regular, df_postseason

def get_team_indices(df):
    teams = pd.unique(df[['Visitor/Neutral', 'Home/Neutral']].values.ravel('K'))
    teams.sort()
    return {team: idx for idx, team in enumerate(teams)}

def create_matrices(df, team_indices):
    wins_matrix = np.zeros((len(team_indices), len(team_indices)))
    games_matrix = np.zeros((len(team_indices), len(team_indices)))
    
    for _, row in df.iterrows():
        home_team = row['Home/Neutral']
        away_team = row['Visitor/Neutral']
        home_pts = row['PTS.1']
        away_pts = row['PTS']
        
        home_idx = team_indices[home_team]
        away_idx = team_indices[away_team]
        
        games_matrix[home_idx, away_idx] += 1
        
        if home_pts > away_pts:
            wins_matrix[home_idx, away_idx] += 1
    
    win_ratio_matrix = np.divide(wins_matrix, games_matrix, out=np.zeros_like(wins_matrix), where=games_matrix != 0)
    return win_ratio_matrix

def prepare_dataset(df, W, H, team_indices):
    games = []
    labels = []
    
    for _, row in df.iterrows():
        home_team = row['Home/Neutral']
        away_team = row['Visitor/Neutral']
        home_pts = row['PTS.1']
        away_pts = row['PTS']
        
        home_idx = team_indices[home_team]
        away_idx = team_indices[away_team]
        
        feature_vector = np.hstack([W[home_idx], H[away_idx]])
        games.append(feature_vector)
        labels.append(1 if home_pts > away_pts else 0)
    
    return np.array(games), np.array(labels)

def main():
    # File paths
    regular_season_file_path = 'regular_games.csv'
    postseason_file_path = 'post_games.csv'
    
    # Load data
    df_regular, df_postseason = load_data(regular_season_file_path, postseason_file_path)
    
    # Get team indices
    team_indices = get_team_indices(df_regular)
    
    # Create win ratio matrices
    win_ratio_matrix_regular = create_matrices(df_regular, team_indices)
    win_ratio_matrix_post = create_matrices(df_postseason, team_indices)
    
    # Apply NMF
    nmf = NMF(n_components=5, init='random', random_state=0)
    W = nmf.fit_transform(win_ratio_matrix_regular)
    H = nmf.components_.T
    
    # Prepare training dataset
    X_train, y_train = prepare_dataset(df_regular, W, H, team_indices)

#     param_grid = {
#     'hidden_layer_sizes': [(50, 10), (100,), (50, 50)],
#     'activation': ['relu', 'tanh'],
#     'solver': ['adam', 'sgd'],
#     'learning_rate': ['constant', 'adaptive'],
#     'max_iter': [50, 200, 300],
#     'alpha': [0.0001, 0.001, 0.01],
#     'tol': [1e-3, 1e-4, 1e-5],
#     'batch_size': [100, 200, 500],
#     'learning_rate_init': [0.1, 0.2, 0.3],
#     'momentum': [0.0, 0.5, 0.9]
# }
    param_grid = {
    'hidden_layer_sizes': [(100,)],
    'activation': ['relu'],
    'solver': ['sgd'],
    'learning_rate': ['constant'],
    'max_iter': [200],
    'alpha': [0.001],
    'tol': [1e-3],
    'batch_size': [200],
    'learning_rate_init': [0.3],
    'momentum': [0.5]
}

    
    # Train classifier
    mlp =  MLPClassifier(random_state=1)
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    
    # Prepare test dataset
    X_test, y_test = prepare_dataset(df_postseason, W, H, team_indices)
    
    # Evaluate model
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

    print(f'Validation AUROC: {grid_search.best_score_}')
    
    print(f'ROC AUC: {roc_auc}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Precision-Recall AUC: {pr_auc}')
    print('Confusion Matrix:')
    print(conf_matrix)

if __name__ == '__main__':
    main()
