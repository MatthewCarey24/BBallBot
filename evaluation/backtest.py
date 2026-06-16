"""Walk-forward backtesting across multiple NBA seasons."""

import os
import sys

# Allow flat imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from betting import test_profit
from features.combined import build_feature_matrix
from features.ratings import EloSystem

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


def tune_season(df_path: str, frac_test: float, n_trials: int) -> dict:
    """Run Optuna on a single season's training portion and return best params.

    Reuses the production ``objective`` so backtest tuning matches how
    ``train_and_evaluate`` selects hyperparameters per year.
    """
    import optuna

    from config import RANDOM_STATE
    from model_trainer import objective, set_random_seeds

    set_random_seeds()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, df_path, frac_test), n_trials=n_trials)
    return study.best_trial.params


def run_single_season(
    df_path: str,
    year: int,
    params: dict,
    frac_test: float,
    starting_wealth: float,
    seed: int = 42,
) -> dict:
    """Train on the training portion of one season CSV and evaluate on the test portion.

    Mirrors the production path: a calibrated MLP (isotonic) over the
    NMF + Elo + form feature matrix.

    Args:
        df_path: Path to the season CSV.
        year: Calendar year label (used only in the returned dict).
        params: Hyperparameters dict (per-year tuned via ``tune_season`` or fixed).
        frac_test: Fraction of games to hold out for testing.
        starting_wealth: Initial bankroll for the Kelly sim.
        seed: Random seed for the MLP.

    Returns:
        Dict with keys: year, seed, accuracy, favorite_accuracy, profit,
        profit_pct, brier_score, n_test_games.
    """
    df = pd.read_csv(df_path)
    train_size = int(len(df) * (1 - frac_test))

    # Build the full feature matrix once (NMF + Elo + form), then split.
    # NMF fits and Elo updates only on the training portion (leakage-safe).
    X_full, y_full = build_feature_matrix(
        df,
        EloSystem(),
        frac_test,
        params["nmf_n_components"],
        params["alpha_H"],
        params["alpha_W"],
    )

    X_train_full = X_full[:train_size]
    y_train_full = y_full[:train_size]
    X_test = X_full[train_size:]
    y_test = y_full[train_size:]

    model = _create_model(params, random_state=seed)
    model.fit(X_train_full, y_train_full.ravel())

    # Calibrate without refitting the base MLP (production parity).
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    except ImportError:
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_train_full, y_train_full.ravel())

    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)

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
        "favorite_accuracy": _favorite_accuracy(df, train_size),
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
    n_trials: int = 0,
) -> pd.DataFrame:
    """Run walk-forward backtest across all seasons and multiple random seeds.

    Each season is treated independently — the model is meant to be trained
    per year — so when ``n_trials > 0`` hyperparameters are tuned separately
    for every season (matching ``train_and_evaluate``). The season's tuned
    params are then reused across the random seeds, which isolate final-model
    initialisation variance.

    Args:
        data_paths: List of CSV paths in chronological order.
        base_params: Fixed hyperparameters used when ``n_trials == 0``
            (defaults to DEFAULT_PARAMS).
        frac_test: Fraction of each season used as the test set.
        starting_wealth: Initial bankroll.
        n_seeds: Number of random seeds to average over.
        n_trials: Optuna trials per season. 0 = skip tuning and use
            ``base_params`` for every season.

    Returns:
        DataFrame with columns: year, seed, accuracy, favorite_accuracy,
        profit, profit_pct, brier_score, n_test_games.
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

        # Tune once per season (per-year training); reuse across seeds.
        if n_trials > 0:
            print(f"Tuning {year} ({n_trials} trials)...")
            season_params = tune_season(path, frac_test, n_trials)
        else:
            season_params = base_params

        for seed in range(n_seeds):
            result = run_single_season(
                df_path=path,
                year=year,
                params=season_params,
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

    for metric in ("accuracy", "favorite_accuracy", "profit_pct", "brier_score"):
        summary[f"{metric}_mean"] = float(results_df[metric].mean())
        summary[f"{metric}_std"] = float(results_df[metric].std())

    # Edge of the model over the always-bet-favorite baseline
    summary["accuracy_edge"] = summary["accuracy_mean"] - summary["favorite_accuracy_mean"]

    # Best-of-year aggregation
    per_year = (
        results_df.groupby("year")[["accuracy", "favorite_accuracy", "profit_pct", "brier_score"]]
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
        results_df.groupby("year")[
            ["accuracy", "favorite_accuracy", "profit_pct", "brier_score", "n_test_games"]
        ]
        .agg(
            {
                "accuracy": "mean",
                "favorite_accuracy": "mean",
                "profit_pct": "mean",
                "brier_score": "mean",
                "n_test_games": "first",
            }
        )
        .reset_index()
    )

    header = (
        f"{'Year':>6}  {'Test Games':>10}  {'Accuracy':>10}  "
        f"{'Favorite':>10}  {'Profit %':>10}  {'Brier':>8}"
    )
    print(header)
    print("-" * len(header))

    for _, row in per_year.iterrows():
        print(
            f"{int(row['year']):>6}  "
            f"{int(row['n_test_games']):>10}  "
            f"{row['accuracy']:>10.4f}  "
            f"{row['favorite_accuracy']:>10.4f}  "
            f"{row['profit_pct']*100:>9.2f}%  "
            f"{row['brier_score']:>8.4f}"
        )

    print(sep)
    summary = summarize_backtest(results_df)
    print(
        f"{'AGGREGATE':>6}  "
        f"{'':>10}  "
        f"{summary['accuracy_mean']:>10.4f}  "
        f"{summary['favorite_accuracy_mean']:>10.4f}  "
        f"{summary['profit_pct_mean']*100:>9.2f}%  "
        f"{summary['brier_score_mean']:>8.4f}"
    )
    acc_std_str = f"±{summary['accuracy_std']:.4f}"
    fav_std_str = f"±{summary['favorite_accuracy_std']:.4f}"
    pct_std_str = f"±{summary['profit_pct_std'] * 100:.2f}%"
    brier_std_str = f"±{summary['brier_score_std']:.4f}"
    print(
        f"{'(±std)':>6}  "
        f"{'':>10}  "
        f"{acc_std_str:>10}  "
        f"{fav_std_str:>10}  "
        f"{pct_std_str:>10}  "
        f"{brier_std_str:>8}"
    )
    print(sep)
    print(f"Model edge over always-bet-favorite: {summary['accuracy_edge']*100:+.2f} pp")
    print(sep)


def _discover_season_paths(data_dir: str) -> list:
    """Return season CSV paths in chronological (year) order."""
    paths = []
    for name in os.listdir(data_dir):
        if name.startswith("odds_data_") and name.endswith(".csv"):
            paths.append(os.path.join(data_dir, name))
    return sorted(paths)


def main() -> None:
    """CLI entry point: run the walk-forward backtest and print the report."""
    import argparse

    parser = argparse.ArgumentParser(description="Walk-forward backtest for BBallBot.")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "odds_data"),
        help="Directory containing odds_data_<year>.csv files.",
    )
    parser.add_argument("--frac-test", type=float, default=0.2, help="Fraction of each season held out for testing.")
    parser.add_argument("--starting-wealth", type=float, default=1000.0, help="Initial bankroll for the Kelly sim.")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds to average over.")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=0,
        help="Optuna trials per season (per-year tuning). 0 uses fixed DEFAULT_PARAMS.",
    )
    args = parser.parse_args()

    data_paths = _discover_season_paths(args.data_dir)
    if not data_paths:
        raise SystemExit(f"No odds_data_<year>.csv files found in {args.data_dir}")

    results = run_walk_forward(
        data_paths,
        frac_test=args.frac_test,
        starting_wealth=args.starting_wealth,
        n_seeds=args.n_seeds,
        n_trials=args.n_trials,
    )
    print_backtest_report(results)


if __name__ == "__main__":
    main()
