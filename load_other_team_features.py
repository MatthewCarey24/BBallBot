import pandas as pd
import os

def load_team_stats():
    data_path = 'odds_data/'
    team_features = pd.read_csv(os.path.join(data_path,'team_info.csv'))
    return team_features, team_features