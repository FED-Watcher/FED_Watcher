import sys
import pandas as pd
import itertools
from pathlib import Path
from datetime import datetime

# ==========================================
# PATH SETUP
# ==========================================
# Get the directory where this script is located (binary_model/)
current_dir = Path(__file__).resolve().parent
# Point to the local 'src' folder (binary_model/src/)
local_src_dir = current_dir / "src"
# Add it to system path so we can import files from it directly
sys.path.append(str(local_src_dir))

# Get Project Root for loading data (4 levels up from run_grid_search.py)
PROJECT_ROOT = current_dir.parent.parent.parent

# ==========================================
# IMPORTS
# ==========================================
try:
    from data_preparation import prepare_pipeline
    from model_training import (
        train_xgboost_model,
        evaluate_model,
        save_model,
        save_metrics,
        get_feature_importance,
    )
    from visualization import generate_plots
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Looking in: {local_src_dir}")
    print(
        "Please ensure data_preparation.py, model_training.py, and visualization.py are in src/models/binary_model/src/"
    )
    sys.exit(1)

# ==========================================
# CONFIGURATION GRIDS
# ==========================================
PARAM_GRID = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200],
    "subsample": [0.8, 1.0],
    "objective": ["binary:logistic"],
    "eval_metric": ["logloss"],
    "random_state": [42],
}


def get_param_combinations(grid):
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def run_experiments():
    DATA_PATH = PROJECT_ROOT / "data" / "MasterDataset_Enriched.csv"
    EXPERIMENTS_DIR = current_dir / "experiments"

    print(f"Starting Grid Search.")
    print(f"Data Source: {DATA_PATH}")
    print(f"Output Dir:  {EXPERIMENTS_DIR}")

    # Check if data exists
    if not DATA_PATH.exists():
        print(f"CRITICAL ERROR: Data file not found at {DATA_PATH}")
        return

    # 1. Prepare Data
    print("Loading and splitting data...")
    X_train, X_test, y_train, y_test, feature_cols = prepare_pipeline(
        filepath=str(DATA_PATH),
        use_announcement_only=True,
        horizon=1,
        test_size=0.25,
        shuffle_data=True,
    )

    combinations = get_param_combinations(PARAM_GRID)
    results_summary = []

    print(f"\nFound {len(combinations)} combinations to test.\n")

    for i, params in enumerate(combinations, 1):
        run_id = f"exp_{i:02d}_d{params['max_depth']}_lr{params['learning_rate']}_n{params['n_estimators']}"
        run_dir = EXPERIMENTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"--- Running {i}/{len(combinations)}: {run_id} ---")

        # 2. Train
        model = train_xgboost_model(X_train, y_train, params=params)

        # 3. Evaluate
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

        # 4. Save Artifacts
        save_model(model, str(run_dir / "model.pkl"))
        save_metrics(metrics, str(run_dir / "metrics.json"))

        # 5. Generate Plots
        importance_df = get_feature_importance(model, feature_cols)
        generate_plots(model, X_train, y_train, X_test, y_test, importance_df, run_dir)

        # 6. Log for Summary
        row = {
            "experiment_id": run_id,
            "test_accuracy": metrics["test"]["accuracy"],
            "test_f1": metrics["test"]["f1_score"],
            "test_precision": metrics["test"]["precision"],
            "train_accuracy": metrics["train"]["accuracy"],
            **params,
        }
        results_summary.append(row)

    # Save Leaderboard
    summary_df = pd.DataFrame(results_summary)
    summary_df["overfitting_gap"] = summary_df["train_accuracy"] - summary_df["test_accuracy"]

    # Sort by Test F1 Score (Best first)
    summary_df = summary_df.sort_values(by="test_f1", ascending=False)

    summary_path = EXPERIMENTS_DIR / "summary_leaderboard.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE")
    print(f"Results saved to: {summary_path}")
    print("=" * 60)

    # ==========================================
    # PRINT OPTIMAL PARAMETERS
    # ==========================================
    if not summary_df.empty:
        best_run = summary_df.iloc[0]

        print(f"\n🏆 BEST PERFORMING MODEL: {best_run['experiment_id']}")
        print(f"   Test F1 Score: {best_run['test_f1']:.4f}")
        print(f"   Test Accuracy: {best_run['test_accuracy']:.4f}")
        print(f"   Overfitting Gap: {best_run['overfitting_gap']:.4f}")

        print("\n👇 OPTIMAL PARAMETERS (Copy these to model_training.py):")
        print("-" * 40)
        print("params = {")
        print(f"    'objective': '{best_run['objective']}',")
        print(f"    'max_depth': {int(best_run['max_depth'])},")
        print(f"    'learning_rate': {best_run['learning_rate']},")
        print(f"    'n_estimators': {int(best_run['n_estimators'])},")
        print(f"    'subsample': {best_run['subsample']},")
        print(f"    'eval_metric': '{best_run['eval_metric']}',")
        print(f"    'random_state': {int(best_run['random_state'])}")
        print("}")
        print("-" * 40)
        print("=" * 60)


if __name__ == "__main__":
    run_experiments()
