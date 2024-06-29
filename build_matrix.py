import pandas as pd
import numpy as np

# Load the CSV data
file_path = 'regular_games.csv'  # Update this with the path to your CSV file
df = pd.read_csv(file_path)

# Initialize lists of teams
teams = pd.unique(df[['Visitor/Neutral', 'Home/Neutral']].values.ravel('K'))
teams.sort()  # Sorting to have a consistent order
team_indices = {team: idx for idx, team in enumerate(teams)}

# Create matrices to store counts of wins and total games
wins_matrix = np.zeros((len(teams), len(teams)))
games_matrix = np.zeros((len(teams), len(teams)))

# Populate the matrices
for _, row in df.iterrows():
    home_team = row['Home/Neutral']
    away_team = row['Visitor/Neutral']
    home_pts = row['PTS.1']
    away_pts = row['PTS']
    
    home_idx = team_indices[home_team]
    away_idx = team_indices[away_team]
    
    # Increment the total games played
    games_matrix[home_idx, away_idx] += 1
    
    # Increment the wins for the home team if they won
    if home_pts > away_pts:
        wins_matrix[home_idx, away_idx] += 1

# Calculate the win ratio matrix
win_ratio_matrix = np.divide(wins_matrix, games_matrix, out=np.zeros_like(wins_matrix), where=games_matrix != 0)

# Convert win ratio matrix to a DataFrame for better readability
win_ratio_df = pd.DataFrame(win_ratio_matrix, index=teams, columns=teams)

# Print or save the win ratio matrix
print(win_ratio_df)
win_ratio_df.to_csv('nba_win_ratio_matrix.csv', index=True)
