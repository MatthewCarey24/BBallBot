"""Rolling form features for BBallBot — computed without data leakage."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Deque

import numpy as np
import pandas as pd

_DATE_FORMAT = "%d %b %Y"
_DEFAULT_REST_DAYS = 3      # used when a team has no prior game this season
_REST_CAP = 7               # cap rest_days at this value


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(str(date_str).strip(), _DATE_FORMAT)


def add_form_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling form features to a game DataFrame without data leakage.

    Features are computed from games *before* the current row, so there is no
    look-ahead bias.

    Added columns:
    - ``home_rest_days``: days since the home team's last game (capped at 7;
      defaults to 3 for the first appearance).
    - ``away_rest_days``: same for the away team.
    - ``home_b2b``: 1 if the home team played yesterday, else 0.
    - ``away_b2b``: 1 if the away team played yesterday, else 0.
    - ``home_rolling_win_pct``: home team's win % over the last *window* games.
    - ``away_rolling_win_pct``: away team's win % over the last *window* games.
    - ``home_rolling_point_diff``: home team's avg point differential over the
      last *window* games (positive = team outscored opponents on average).
    - ``away_rolling_point_diff``: same for the away team.

    The input DataFrame must have columns:
    ``Date``, ``Home Team``, ``Away Team``, ``Home Score``, ``Away Score``.
    Date strings should be in ``'%d %b %Y'`` format (e.g. ``'29 Oct 2021'``).

    Args:
        df: Game-level DataFrame sorted chronologically.
        window: Rolling window size in games (default 10).

    Returns:
        New DataFrame (copy of *df*) with the eight additional columns.
    """
    df = df.copy().reset_index(drop=True)

    # Per-team state tracked as we iterate chronologically
    last_game_date: dict[str, datetime | None] = defaultdict(lambda: None)

    # Deques of (won: bool, point_diff: float) for the last *window* games
    recent_results: dict[str, Deque[tuple[bool, float]]] = defaultdict(lambda: deque(maxlen=window))

    # Output lists
    home_rest: list[float] = []
    away_rest: list[float] = []
    home_b2b: list[int] = []
    away_b2b: list[int] = []
    home_win_pct: list[float] = []
    away_win_pct: list[float] = []
    home_pt_diff: list[float] = []
    away_pt_diff: list[float] = []

    for _, row in df.iterrows():
        home_team = str(row["Home Team"])
        away_team = str(row["Away Team"])
        game_date = _parse_date(row["Date"])

        home_score = float(row["Home Score"])
        away_score = float(row["Away Score"])

        # ---- Rest days (BEFORE updating state) ----
        def _rest(team: str) -> float:
            prev = last_game_date[team]
            if prev is None:
                return float(_DEFAULT_REST_DAYS)
            return min((game_date - prev).days, _REST_CAP)

        h_rest = _rest(home_team)
        a_rest = _rest(away_team)
        home_rest.append(h_rest)
        away_rest.append(a_rest)
        home_b2b.append(1 if h_rest == 1 else 0)
        away_b2b.append(1 if a_rest == 1 else 0)

        # ---- Rolling win% and point-diff (BEFORE updating state) ----
        def _win_pct(team: str) -> float:
            history = recent_results[team]
            if not history:
                return 0.5  # neutral prior when no games yet
            return sum(1 for won, _ in history if won) / len(history)

        def _pt_diff(team: str) -> float:
            history = recent_results[team]
            if not history:
                return 0.0
            return float(np.mean([d for _, d in history]))

        home_win_pct.append(_win_pct(home_team))
        away_win_pct.append(_win_pct(away_team))
        home_pt_diff.append(_pt_diff(home_team))
        away_pt_diff.append(_pt_diff(away_team))

        # ---- Update state AFTER recording features ----
        home_won = home_score > away_score
        recent_results[home_team].append((home_won, home_score - away_score))
        recent_results[away_team].append((not home_won, away_score - home_score))
        last_game_date[home_team] = game_date
        last_game_date[away_team] = game_date

    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest
    df["home_b2b"] = home_b2b
    df["away_b2b"] = away_b2b
    df["home_rolling_win_pct"] = home_win_pct
    df["away_rolling_win_pct"] = away_win_pct
    df["home_rolling_point_diff"] = home_pt_diff
    df["away_rolling_point_diff"] = away_pt_diff

    return df
