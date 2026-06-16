"""Characterization tests for the betting module."""

import os
import sys

# Ensure the project root is on the path so flat imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from betting import (
    calculate_frac_wealth,
    calculate_implied_proba,
    roi,
    simulate_bankroll,
    vig_free_prob,
)

# ---------------------------------------------------------------------------
# calculate_implied_proba
# ---------------------------------------------------------------------------

class TestCalculateImpliedProba:
    def test_favorite_negative_odds(self):
        """Favorite at -150 should yield ~0.60 implied probability."""
        prob = calculate_implied_proba(-150)
        assert abs(prob - 0.6) < 0.01

    def test_underdog_positive_odds(self):
        """Underdog at +200 should yield ~0.333 implied probability."""
        prob = calculate_implied_proba(200)
        assert abs(prob - 1 / 3) < 0.01

    def test_even_money(self):
        """Even money +100 should yield 0.5 implied probability."""
        prob = calculate_implied_proba(100)
        assert abs(prob - 0.5) < 0.01

    def test_output_in_unit_interval(self):
        for odds in [-500, -200, -110, 100, 110, 200, 500]:
            p = calculate_implied_proba(odds)
            assert 0.0 < p < 1.0, f"Probability {p} out of range for odds {odds}"


# ---------------------------------------------------------------------------
# calculate_frac_wealth
# ---------------------------------------------------------------------------

class TestCalculateFracWealth:
    def _make_proba(self, p_home: float) -> np.ndarray:
        """Build a one-row y_proba array [[p_away, p_home]]."""
        return np.array([[1 - p_home, p_home]])

    def test_returns_non_negative(self):
        """Kelly fraction must always be >= 0."""
        y_proba = self._make_proba(0.70)
        frac = calculate_frac_wealth(-110, 110, y_proba, 0)
        assert frac >= 0

    def test_zero_when_edge_is_negative(self):
        """When the max model probability implies no edge, Kelly returns 0.

        calculate_frac_wealth uses max(p[0], p[1]) as proba_win, so we need a
        scenario where even the higher probability yields a negative Kelly
        fraction.  With odds=-500 (break-even ≈ 0.833) and max prob=0.60,
        the raw Kelly is negative and is clamped to 0.
        """
        # max prob = 0.60, break-even for -500 ≈ 0.833 → negative edge
        y_proba = self._make_proba(0.60)
        frac = calculate_frac_wealth(-500, 500, y_proba, 0)
        assert frac == 0.0

    def test_kelly_cap_never_exceeds_one(self):
        """Fraction returned must never exceed 1.0 (Phase 2 fix: max_fraction cap)."""
        y_proba = self._make_proba(0.99)
        frac = calculate_frac_wealth(100, -100, y_proba, 0)
        assert frac <= 1.0

    def test_kelly_fraction_parameter_scales_output(self):
        """Halving kelly_fraction should halve the output when below the max_fraction cap.

        We choose odds and proba so the raw Kelly * kelly_fraction stays below
        the default max_fraction=0.05, ensuring the cap does not interfere.
        """
        # proba_win = max(0.35, 0.65) = 0.65; odds = +100 → percent_gain = 1.0
        # raw Kelly = 0.65 - 0.35/1.0 = 0.30
        # with kelly_fraction=0.1 → 0.03  (< 0.05 cap)
        # with kelly_fraction=0.05 → 0.015 (< 0.05 cap)
        y_proba = self._make_proba(0.65)
        frac_full = calculate_frac_wealth(100, -100, y_proba, 0, kelly_fraction=0.10, max_fraction=0.05)
        frac_half = calculate_frac_wealth(100, -100, y_proba, 0, kelly_fraction=0.05, max_fraction=0.05)
        if frac_full > 0:
            assert abs(frac_half - frac_full * 0.5) < 1e-9

    def test_max_fraction_parameter_caps_output(self):
        """max_fraction should act as a hard ceiling."""
        y_proba = self._make_proba(0.99)
        cap = 0.02
        frac = calculate_frac_wealth(100, -100, y_proba, 0, kelly_fraction=1.0, max_fraction=cap)
        assert frac <= cap


# ---------------------------------------------------------------------------
# vig_free_prob
# ---------------------------------------------------------------------------

class TestVigFreeProb:
    def test_two_sides_sum_to_one(self):
        """The de-vigged probabilities of both sides must sum to 1."""
        home = vig_free_prob(-110, -110)
        away = vig_free_prob(-110, -110)
        assert home + away == pytest.approx(1.0)

    def test_equal_odds_gives_half(self):
        """Symmetric -110/-110 line is a true coin flip after removing vig."""
        assert vig_free_prob(-110, -110) == pytest.approx(0.5)

    def test_is_below_raw_implied(self):
        """De-vigged probability is lower than the raw implied (vig removed)."""
        raw = calculate_implied_proba(-150)
        novig = vig_free_prob(-150, 130)
        assert novig < raw


# ---------------------------------------------------------------------------
# calculate_frac_wealth — edge gate
# ---------------------------------------------------------------------------

class TestEdgeThreshold:
    def _make_proba(self, p_home: float) -> np.ndarray:
        return np.array([[1 - p_home, p_home]])

    def test_no_bet_when_edge_below_threshold(self):
        """Model barely above the market should not clear a 10% threshold."""
        # -110/-110 → market 0.5; model 0.55 → edge 0.05 < 0.10
        y_proba = self._make_proba(0.55)
        frac = calculate_frac_wealth(-110, -110, y_proba, 0, edge_threshold=0.10)
        assert frac == 0.0

    def test_bets_when_edge_clears_threshold(self):
        """Model well above the market should bet through the threshold."""
        # -110/-110 → market 0.5; model 0.70 → edge 0.20 > 0.10
        y_proba = self._make_proba(0.70)
        frac = calculate_frac_wealth(-110, -110, y_proba, 0, edge_threshold=0.10)
        assert frac > 0.0


# ---------------------------------------------------------------------------
# simulate_bankroll
# ---------------------------------------------------------------------------

class TestSimulateBankroll:
    def test_winning_value_bet_grows_bankroll(self):
        odds_df = pd.DataFrame({"Home Odds": [100], "Away Odds": [-100]})
        y_pred = np.array([1])
        y_true = np.array([1])
        y_proba = np.array([[0.2, 0.8]])  # confident home, market 0.5 → edge
        wealth, stake = simulate_bankroll(odds_df, y_pred, y_true, y_proba, 1000.0)
        assert wealth > 1000.0
        assert stake > 0.0

    def test_no_edge_means_no_stake(self):
        odds_df = pd.DataFrame({"Home Odds": [-110], "Away Odds": [-110]})
        y_pred = np.array([1])
        y_true = np.array([1])
        y_proba = np.array([[0.48, 0.52]])  # below market+threshold
        wealth, stake = simulate_bankroll(
            odds_df, y_pred, y_true, y_proba, 1000.0, edge_threshold=0.05
        )
        assert stake == 0.0
        assert wealth == 1000.0


# ---------------------------------------------------------------------------
# Payout calculation (inline, not a separate function — test the arithmetic)
# ---------------------------------------------------------------------------

class TestPayoutCalculation:
    def test_positive_odds_payout(self):
        """Positive odds +150 on a $100 bet should net $150."""
        odds = 150
        bet = 100.0
        winnings = (odds / 100) * bet
        assert winnings == pytest.approx(150.0)

    def test_negative_odds_payout(self):
        """Negative odds -200 on a $100 bet should net $50."""
        odds = -200
        bet = 100.0
        winnings = (100 / abs(odds)) * bet
        assert winnings == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# roi helper
# ---------------------------------------------------------------------------

class TestRoi:
    def test_basic_roi(self):
        assert roi(200.0, 1000.0) == pytest.approx(0.2)

    def test_negative_roi(self):
        assert roi(-50.0, 1000.0) == pytest.approx(-0.05)

    def test_zero_profit(self):
        assert roi(0.0, 1000.0) == pytest.approx(0.0)
