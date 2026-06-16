"""Elo rating system with season-to-season carryover for BBallBot."""

import math

import pandas as pd

DEFAULT_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 100.0  # Elo points added to home team expected score


class EloSystem:
    """Maintains and updates Elo ratings for NBA teams.

    Ratings persist across ``process_season`` calls, enabling season-to-season
    carryover.  Call ``decay_toward_mean`` between seasons to apply regression
    to the mean.
    """

    def __init__(
        self,
        k: float = K_FACTOR,
        home_adv: float = HOME_ADVANTAGE,
        default: float = DEFAULT_ELO,
    ) -> None:
        self.k = k
        self.home_adv = home_adv
        self.default = default
        self.ratings: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_rating(self, team: str) -> float:
        """Return current Elo rating for *team*, initialising to default if new."""
        return self.ratings.setdefault(team, self.default)

    def expected_home_win_prob(self, home_team: str, away_team: str) -> float:
        """Logistic win probability for the home team.

        Formula: ``1 / (1 + 10 ** ((away_elo - home_elo - home_adv) / 400))``
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)
        exponent = (away_elo - home_elo - self.home_adv) / 400.0
        return 1.0 / (1.0 + math.pow(10.0, exponent))

    def update(self, home_team: str, away_team: str, home_won: bool) -> None:
        """Update Elo ratings after a game result.

        Args:
            home_team: Name of the home team.
            away_team: Name of the away team.
            home_won: True if the home team won.
        """
        expected = self.expected_home_win_prob(home_team, away_team)
        actual = 1.0 if home_won else 0.0

        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        self.ratings[home_team] = home_elo + self.k * (actual - expected)
        self.ratings[away_team] = away_elo + self.k * (expected - actual)

    def get_pre_game_features(self, home_team: str, away_team: str) -> dict:
        """Return a dict of pre-game Elo features for the matchup.

        Returns:
            Dict with keys ``home_elo``, ``away_elo``, ``elo_diff``,
            ``home_win_prob``.
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)
        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "elo_win_prob": self.expected_home_win_prob(home_team, away_team),
        }

    def process_season(self, df: pd.DataFrame, frac_test: float = 0.0) -> pd.DataFrame:
        """Process all games chronologically, adding pre-game Elo feature columns.

        Ratings are updated only for training games (the first
        ``1 - frac_test`` fraction of the DataFrame).  Test-split games get Elo
        features recorded but do NOT cause rating updates, preventing leakage.

        Ratings persist after the call so they carry over to subsequent seasons.

        Args:
            df: Game DataFrame with columns ``Home Team``, ``Away Team``,
                ``Home Score``, ``Away Score``.  Must be sorted chronologically.
            frac_test: Fraction of rows to treat as test (no rating updates).

        Returns:
            New DataFrame (copy of *df*) with added columns:
            ``home_elo``, ``away_elo``, ``elo_diff``, ``elo_win_prob``.
        """
        df = df.copy().reset_index(drop=True)
        train_cutoff = int(len(df) * (1 - frac_test))

        home_elos = []
        away_elos = []
        elo_diffs = []
        win_probs = []

        for i, row in df.iterrows():
            home_team = str(row["Home Team"])
            away_team = str(row["Away Team"])

            # Record pre-game features BEFORE updating ratings
            feats = self.get_pre_game_features(home_team, away_team)
            home_elos.append(feats["home_elo"])
            away_elos.append(feats["away_elo"])
            elo_diffs.append(feats["elo_diff"])
            win_probs.append(feats["elo_win_prob"])

            # Only update ratings in the training portion
            if i < train_cutoff:
                home_won = float(row["Home Score"]) > float(row["Away Score"])
                self.update(home_team, away_team, home_won)

        df["home_elo"] = home_elos
        df["away_elo"] = away_elos
        df["elo_diff"] = elo_diffs
        df["elo_win_prob"] = win_probs
        return df

    def decay_toward_mean(self, fraction: float = 0.33) -> None:
        """Pull all ratings ``fraction`` of the way toward the current mean.

        Call this between seasons to apply regression to the mean.

        Args:
            fraction: How far to move toward the mean (0 = no change, 1 = reset
                everything to the mean).  Default 0.33 (one-third regression).
        """
        if not self.ratings:
            return
        mean = sum(self.ratings.values()) / len(self.ratings)
        for team in self.ratings:
            self.ratings[team] += fraction * (mean - self.ratings[team])
