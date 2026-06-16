"""Integration module: combine NMF, Elo, and form features into one matrix."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data_processor import create_features
from features.form import add_form_features
from features.ratings import EloSystem

# Form feature column names produced by add_form_features
_FORM_COLS = [
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    "home_rolling_win_pct",
    "away_rolling_win_pct",
    "home_rolling_point_diff",
    "away_rolling_point_diff",
]

# Elo feature column names produced by EloSystem.process_season
_ELO_COLS = ["home_elo", "away_elo", "elo_diff", "elo_win_prob"]


def build_feature_matrix(
    df: pd.DataFrame,
    elo_system: "EloSystem",
    frac_test: float,
    nmf_n_components: int,
    nmf_alpha_H: float,
    nmf_alpha_W: float,
    use_elo: bool = True,
    use_form: bool = True,
    use_nmf: bool = True,
    form_window: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the full feature matrix by combining selected feature blocks.

    Feature blocks (in order when all enabled):
    1. NMF latent factors + implied-probability odds (from ``create_features``).
    2. Elo ratings: home_elo, away_elo, elo_diff, elo_win_prob.
    3. Form features: rest days, back-to-back flags, rolling win%, rolling
       point differential.

    Args:
        df: Season game DataFrame in chronological order.
        elo_system: An ``EloSystem`` instance (ratings carry over if called
            sequentially across seasons).
        frac_test: Fraction of games to treat as test (used for NMF win-loss
            matrix and Elo update boundary).
        nmf_n_components: NMF rank.
        nmf_alpha_H: L2 regularisation on H.
        nmf_alpha_W: L2 regularisation on W.
        use_elo: Include Elo features.
        use_form: Include rolling form features.
        use_nmf: Include NMF + odds features.
        form_window: Rolling window (in games) for form features.

    Returns:
        ``(X, y)`` where ``X`` is shape ``(n_games, n_features)`` and ``y`` is
        binary (1 = home win).
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)

    feature_blocks: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Block 1 — NMF latent factors + implied-probability odds
    # ------------------------------------------------------------------
    if use_nmf:
        X_nmf, y = create_features(df, frac_test, nmf_n_components, nmf_alpha_H, nmf_alpha_W)
        feature_blocks.append(X_nmf)
    else:
        # Still need y labels
        y = np.array(
            [1 if float(row["Home Score"]) > float(row["Away Score"]) else 0 for _, row in df.iterrows()],
            dtype=np.int64,
        )

    # ------------------------------------------------------------------
    # Block 2 — Elo ratings
    # ------------------------------------------------------------------
    if use_elo:
        df_elo = elo_system.process_season(df, frac_test=frac_test)
        elo_arr = df_elo[_ELO_COLS].values.astype(np.float64)
        # Normalise Elo values to roughly the same scale as other features
        # (divide by 1500 so ratings ≈ 1.0 at default, diffs ≈ 0)
        elo_arr[:, :2] /= 1500.0  # home_elo, away_elo
        elo_arr[:, 2] /= 400.0    # elo_diff (one "sigma" ≈ 400 pts)
        # elo_win_prob already in [0, 1]
        feature_blocks.append(elo_arr)

    # ------------------------------------------------------------------
    # Block 3 — Rolling form features
    # ------------------------------------------------------------------
    if use_form:
        df_form = add_form_features(df, window=form_window)
        form_arr = df_form[_FORM_COLS].values.astype(np.float64)
        feature_blocks.append(form_arr)

    # Concatenate all blocks
    if feature_blocks:
        X = np.hstack(feature_blocks)
    else:
        # Degenerate case: no features requested — return empty matrix
        X = np.empty((n, 0), dtype=np.float64)

    return X, y
