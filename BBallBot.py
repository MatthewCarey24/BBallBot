import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
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
    
    # Train classifier
    clf = GradientBoostingClassifier(n_estimators=25, learning_rate=0.01, max_depth=4, subsample=1.0)
    clf.fit(X_train, y_train)
    
    # Prepare test dataset
    X_test, y_test = prepare_dataset(df_postseason, W, H, team_indices)
    
    # Evaluate model
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
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
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Precision-Recall AUC: {pr_auc}')
    print('Confusion Matrix:')
    print(conf_matrix)

if __name__ == '__main__':
    main()
