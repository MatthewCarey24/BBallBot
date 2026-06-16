"""Walk-forward backtesting across multiple NBA seasons."""

import os
import sys

# Allow flat imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from betting import test_profit
from data_processor import create_features

# ---------------------------------------------------------------------------
# Default fixed hyperparameters (used in backtest to skip costly Optuna)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "nmf_n_components": 6,
    "alpha_H": 0.0501,
    "alpha_W": 0.0501,
    "first_layer_neurons": 20,
    "activation": "tanh",
    "solver": "adam",
    "alpha": 1.0,
    "learning_rate": "adaptive",
    "learning_rate_init": 0.001,
}


def _create_model(params: dict, random_state: int = 42) -> Pipeline:
    """Build a StandardScaler + MLPClassifier pipeline from *params*."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(params["first_layer_neurons"],),
            activation=params["activation"],
            solver=params["solver"],
            alpha=params["alpha"],
            learning_rate=params["learning_rate"],
            learning_rate_init=params["learning_rate_init"],
            max_iter=20000,
            random_state=random_state,
        )),
    ])


def _favorite_accuracy(df: pd.DataFrame, start_idx: int) -> float:
    """Compute the fraction of test games won by the moneyline favourite.

    In NBA moneyline data the favourite is the team with the more negative
    (or less positive) odds.  A home-team win is label 1.
    """
    df_test = df.iloc[start_idx:].copy()
    correct = 0
    total = 0
    for _, row in df_test.iterrows():
        try:
            h_odds = float(row["Home Odds"])
            a_odds = float(row["Away Odds"])
        except (ValueError, TypeError):
            continue
        # Favourite: lower (more negative) odds → higher implied probability
        home_is_fav = h_odds < a_odds
        home_won = float(row["Home Score"]) > float(row["Away Score"])
        if home_is_fav == home_won:
            correct += 1
        total += 1
    return correct / total if total > 0 else float("nan")


def run_single_season(
    df_path: str,
    year: int,
    params: dict,
    frac_test: float,
    starting_wealth: float,
    seed: int = 42,
) -> dict:
    """Train on the training portion of one season CSV and evaluate on the test portion.

    Args:
        df_path: Path to the season CSV.
        year: Calendar year label (used only in the returned dict).
        params: Fixed hyperparameters dict (skip Optuna for speed).
        frac_test: Fraction of games to hold out for testing.
        starting_wealth: Initial bankroll for the Kelly sim.
        seed: Random seed for the MLP.

    Returns:
        Dict with keys: year, seed, accuracy, profit, profit_pct, brier_score, n_test_games.
    """
    df = pd.read_csv(df_path)
    train_size = int(len(df) * (1 - frac_test))
    df_train = df.iloc[:train_size]

    # Build NMF features on training portion only
    X_train_full, y_train_full = create_features(
        df_train,
        0.0,  # use all training rows
        params["nmf_n_components"],
        params["alpha_H"],
        params["alpha_W"],
    )

    # Build features for full dataset to get the test split
    X_full, y_full = create_features(
        df,
        frac_test,
        params["nmf_n_components"],
        params["alpha_H"],
        params["alpha_W"],
    )

    X_test = X_full[train_size:]
    y_test = y_full[train_size:]

    model = _create_model(params, random_state=seed)
    model.fit(X_train_full, y_train_full.ravel())

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    brier = float(brier_score_loss(y_test, y_proba[:, 1]))

    import sys as _sys
    from io import StringIO as _StringIO
    _old_stdout = _sys.stdout
    _sys.stdout = _StringIO()
    try:
        wealth, total_stake = test_profit(df_path, y_pred, y_test, y_proba, starting_wealth, frac_test)
    finally:
        _sys.stdout = _old_stdout

    profit = float(wealth - starting_wealth)
    profit_pct = float(profit / total_stake) if total_stake > 0 else 0.0

    return {
        "year": year,
        "seed": seed,
        "accuracy": accuracy,
        "profit": profit,
        "profit_pct": profit_pct,
        "brier_score": brier,
        "n_test_games": int(len(y_test)),
    }


def run_walk_forward(
    data_paths: list,
    base_params: dict | None = None,
    frac_test: float = 0.2,
    starting_wealth: float = 1000.0,
    n_seeds: int = 5,
) -> pd.DataFrame:
    """Run walk-forward backtest across all seasons and multiple random seeds.

    Args:
        data_paths: List of CSV paths in chronological order.
        base_params: Fixed hyperparameters (defaults to DEFAULT_PARAMS).
        frac_test: Fraction of each season used as the test set.
        starting_wealth: Initial bankroll.
        n_seeds: Number of random seeds to average over.

    Returns:
        DataFrame with columns: year, seed, accuracy, profit, profit_pct, brier_score,
        n_test_games.
    """
    if base_params is None:
        base_params = DEFAULT_PARAMS

    records = []
    for path in data_paths:
        # Infer year from filename (e.g. odds_data_2022.csv → 2022)
        basename = os.path.basename(path)
        try:
            year = int(basename.replace("odds_data_", "").replace(".csv", ""))
        except ValueError:
            year = 0

        for seed in range(n_seeds):
            result = run_single_season(
                df_path=path,
                year=year,
                params=base_params,
                frac_test=frac_test,
                starting_wealth=starting_wealth,
                seed=seed,
            )
            records.append(result)

    return pd.DataFrame(records)


def summarize_backtest(results_df: pd.DataFrame) -> dict:
    """Compute aggregate statistics across seeds.

    Returns:
        Dict with mean ± std for accuracy, profit_pct, brier_score, plus
        a per-season always_bet_favorite_accuracy baseline.
    """
    summary: dict = {}

    for metric in ("accuracy", "profit_pct", "brier_score"):
        summary[f"{metric}_mean"] = float(results_df[metric].mean())
        summary[f"{metric}_std"] = float(results_df[metric].std())

    # Best-of-year aggregation
    per_year = (
        results_df.groupby("year")[["accuracy", "profit_pct", "brier_score"]]
        .mean()
        .reset_index()
    )
    summary["per_year"] = per_year.to_dict(orient="records")

    return summary


def print_backtest_report(results_df: pd.DataFrame) -> None:
    """Print a formatted comparison table of per-season and aggregate results."""
    sep = "-" * 70
    print(sep)
    print(f"{'WALK-FORWARD BACKTEST REPORT':^70}")
    print(sep)

    per_year = (
        results_df.groupby("year")[["accuracy", "profit_pct", "brier_score", "n_test_games"]]
        .agg({"accuracy": "mean", "profit_pct": "mean", "brier_score": "mean", "n_test_games": "first"})
        .reset_index()
    )

    header = f"{'Year':>6}  {'Test Games':>10}  {'Accuracy':>10}  {'Profit %':>10}  {'Brier':>8}"
    print(header)
    print("-" * len(header))

    for _, row in per_year.iterrows():
        print(
            f"{int(row['year']):>6}  "
            f"{int(row['n_test_games']):>10}  "
            f"{row['accuracy']:>10.4f}  "
            f"{row['profit_pct']*100:>9.2f}%  "
            f"{row['brier_score']:>8.4f}"
        )

    print(sep)
    summary = summarize_backtest(results_df)
    print(
        f"{'AGGREGATE':>6}  "
        f"{'':>10}  "
        f"{summary['accuracy_mean']:>10.4f}  "
        f"{summary['profit_pct_mean']*100:>9.2f}%  "
        f"{summary['brier_score_mean']:>8.4f}"
    )
    acc_std_str = f"±{summary['accuracy_std']:.4f}"
    pct_std_str = f"±{summary['profit_pct_std'] * 100:.2f}%"
    brier_std_str = f"±{summary['brier_score_std']:.4f}"
    print(
        f"{'(±std)':>6}  "
        f"{'':>10}  "
        f"{acc_std_str:>10}  "
        f"{pct_std_str:>10}  "
        f"{brier_std_str:>8}"
    )
    print(sep)
