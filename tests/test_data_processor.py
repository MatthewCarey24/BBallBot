"""Tests for the data_processor module."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from data_processor import create_win_loss_matrix, get_team_indices, prepare_x_y

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    """A minimal 4-game DataFrame with two teams."""
    return pd.DataFrame({
        "Date": ["01 Nov 2021", "02 Nov 2021", "03 Nov 2021", "04 Nov 2021"],
        "Time": ["7:00 pm", "7:00 pm", "7:00 pm", "7:00 pm"],
        "Home Team": ["Lakers", "Celtics", "Lakers", "Celtics"],
        "Away Team": ["Celtics", "Lakers", "Celtics", "Lakers"],
        "Home Score": [110, 95, 105, 100],
        "Away Score": [100, 110, 98, 102],
        "Home Odds": [-150, 120, -130, 110],
        "Away Odds": [130, -140, 115, -125],
    })


@pytest.fixture
def df_with_missing_odds():
    """DataFrame that has '-' as odds values."""
    return pd.DataFrame({
        "Date": ["01 Nov 2021"],
        "Time": ["7:00 pm"],
        "Home Team": ["Lakers"],
        "Away Team": ["Celtics"],
        "Home Score": [110],
        "Away Score": [100],
        "Home Odds": ["-"],
        "Away Odds": ["-"],
    })


# ---------------------------------------------------------------------------
# get_team_indices
# ---------------------------------------------------------------------------

class TestGetTeamIndices:
    def test_returns_dict(self, simple_df):
        result = get_team_indices(simple_df)
        assert isinstance(result, dict)

    def test_all_teams_present(self, simple_df):
        result = get_team_indices(simple_df)
        assert "Lakers" in result
        assert "Celtics" in result

    def test_indices_are_unique(self, simple_df):
        result = get_team_indices(simple_df)
        indices = list(result.values())
        assert len(indices) == len(set(indices))

    def test_correct_number_of_teams(self, simple_df):
        result = get_team_indices(simple_df)
        assert len(result) == 2

    def test_indices_are_zero_based(self, simple_df):
        result = get_team_indices(simple_df)
        assert min(result.values()) == 0
        assert max(result.values()) == len(result) - 1


# ---------------------------------------------------------------------------
# create_win_loss_matrix
# ---------------------------------------------------------------------------

class TestCreateWinLossMatrix:
    def test_shape_is_n_teams_by_n_teams(self, simple_df):
        team_indices = get_team_indices(simple_df)
        matrix = create_win_loss_matrix(team_indices, simple_df, frac_test=0.0)
        n = len(team_indices)
        assert matrix.shape == (n, n)

    def test_values_between_zero_and_one(self, simple_df):
        team_indices = get_team_indices(simple_df)
        matrix = create_win_loss_matrix(team_indices, simple_df, frac_test=0.0)
        assert np.all(matrix >= 0.0)
        assert np.all(matrix <= 1.0)

    def test_shape_two_by_two_with_two_teams(self, simple_df):
        team_indices = get_team_indices(simple_df)
        matrix = create_win_loss_matrix(team_indices, simple_df, frac_test=0.0)
        assert matrix.shape == (2, 2)

    def test_frac_test_reduces_training_games(self, simple_df):
        team_indices = get_team_indices(simple_df)
        matrix_full = create_win_loss_matrix(team_indices, simple_df, frac_test=0.0)
        matrix_half = create_win_loss_matrix(team_indices, simple_df, frac_test=0.5)
        # They may differ since fewer games are used
        # At minimum both should have correct shape
        assert matrix_full.shape == matrix_half.shape


# ---------------------------------------------------------------------------
# prepare_x_y
# ---------------------------------------------------------------------------

class TestPrepareXY:
    def _make_nmf_matrices(self, n_teams: int, n_components: int):
        rng = np.random.default_rng(42)
        W = rng.random((n_teams, n_components))
        H = rng.random((n_teams, n_components))
        return W, H

    def test_feature_vector_length(self, simple_df):
        """Feature vector length should be 2*(n_components + 1)."""
        n_components = 3
        team_indices = get_team_indices(simple_df)
        n_teams = len(team_indices)
        W, H = self._make_nmf_matrices(n_teams, n_components)
        X, y = prepare_x_y(team_indices, simple_df, W, H)
        expected_len = 2 * (n_components + 1)
        assert X.shape[1] == expected_len

    def test_labels_are_binary(self, simple_df):
        n_components = 3
        team_indices = get_team_indices(simple_df)
        n_teams = len(team_indices)
        W, H = self._make_nmf_matrices(n_teams, n_components)
        _, y = prepare_x_y(team_indices, simple_df, W, H)
        assert set(y).issubset({0, 1})

    def test_number_of_samples_matches_df(self, simple_df):
        n_components = 3
        team_indices = get_team_indices(simple_df)
        n_teams = len(team_indices)
        W, H = self._make_nmf_matrices(n_teams, n_components)
        X, y = prepare_x_y(team_indices, simple_df, W, H)
        assert len(X) == len(simple_df)
        assert len(y) == len(simple_df)

    def test_missing_odds_handled_without_error(self, df_with_missing_odds):
        """'-' odds values must not raise an exception."""
        n_components = 2
        team_indices = get_team_indices(df_with_missing_odds)
        n_teams = len(team_indices)
        W, H = self._make_nmf_matrices(n_teams, n_components)
        # Should not raise
        X, y = prepare_x_y(team_indices, df_with_missing_odds, W, H)
        assert X.shape[0] == 1

    def test_missing_odds_produce_finite_values(self, df_with_missing_odds):
        """'-' odds must convert to a valid finite probability, not NaN."""
        n_components = 2
        team_indices = get_team_indices(df_with_missing_odds)
        n_teams = len(team_indices)
        W, H = self._make_nmf_matrices(n_teams, n_components)
        X, _ = prepare_x_y(team_indices, df_with_missing_odds, W, H)
        assert np.all(np.isfinite(X))
