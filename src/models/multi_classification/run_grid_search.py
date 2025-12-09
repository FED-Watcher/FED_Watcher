import sys
import pandas as pd
import itertools
import shutil
import numpy as np
from pathlib import Path

# ==========================================
# PATH SETUP
# ==========================================
current_dir = Path(__file__).resolve().parent
local_src_dir = current_dir / "src"
sys.path.append(str(local_src_dir))
PROJECT_ROOT = current_dir.parent.parent.parent

# ==========================================
# IMPORTS
# ==========================================
try:
    from data_preparation import load_raw_data, prepare_data_with_thresholds
    from model_training import train_model, evaluate_model
    from visualization import generate_plots
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# ==========================================
# 1. THRESHOLD CONFIGURATIONS
# ==========================================
# We test different definitions of "Market Moves"
THRESHOLD_GRIDS = {
    # Original: Standard thresholds
    "Standard": {"strong": 1.5, "modest": 0.5},
    # Sensitive: Detects smaller movements (Good if market is low volatility)
    "Sensitive": {"strong": 1.0, "modest": 0.3},
    # Stable: Ignores noise, focuses on big moves (Good if market is noisy)
    "Stable": {"strong": 2.0, "modest": 0.8},
}

# ==========================================
# 2. MODEL HYPERPARAMETERS
# ==========================================
PARAM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8],
    "max_features": ["sqrt"],
    "random_state": [42],
}


def get_param_combinations(grid):
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def run_experiments():
    DATA_PATH = PROJECT_ROOT / "data" / "MasterDataset_Enriched.csv"
    EXPERIMENTS_DIR = current_dir / "experiments"

    if EXPERIMENTS_DIR.exists():
        shutil.rmtree(EXPERIMENTS_DIR)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting Advanced Grid Search (Thresholds + Models).")

    # 1. Load Raw Data (Once)
    df_raw = load_raw_data(str(DATA_PATH))

    model_combinations = get_param_combinations(PARAM_GRID)
    results = []

    total_runs = len(THRESHOLD_GRIDS) * len(model_combinations)
    run_counter = 0

    # ==========================================
    # OUTER LOOP: THRESHOLDS
    # ==========================================
    for thresh_name, thresh_vals in THRESHOLD_GRIDS.items():
        print(
            f"\n[Threshold Config: {thresh_name}] Strong: {thresh_vals['strong']}%, Modest: {thresh_vals['modest']}%"
        )

        # Re-generate Target Variable based on new thresholds
        X_train, X_test, y_train, y_test, _, weights, features = prepare_data_with_thresholds(
            df_raw, thresh_vals
        )

        # Calculate Class Distribution (To see if a threshold is unbalanced)
        unique, counts = np.unique(y_test, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"  Test Class Dist: {class_dist}")

        # ==========================================
        # INNER LOOP: MODEL PARAMS
        # ==========================================
        for params in model_combinations:
            run_counter += 1

            # Create ID: e.g., "Sensitive_lr0.05_d3"
            run_id = f"{thresh_name}_lr{params['learning_rate']}_d{params['max_depth']}_n{params['n_estimators']}"
            run_dir = EXPERIMENTS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"  > ({run_counter}/{total_runs}) Training {run_id}...", end="\r")

            # Train
            model = train_model(X_train, y_train, weights, params)

            # Evaluate
            metrics, _ = evaluate_model(model, X_test, y_test)

            # Generate Plots
            generate_plots(model, X_test, y_test, features, run_dir)

            # Log Data
            row = {
                "experiment_id": run_id,
                "threshold_mode": thresh_name,
                "thresh_strong": thresh_vals["strong"],
                "thresh_modest": thresh_vals["modest"],
                **metrics,
                **params,
            }
            results.append(row)

    print("\n\nExperiments complete.")

    # ==========================================
    # RESULTS ANALYSIS
    # ==========================================
    df_res = pd.DataFrame(results)

    # Sort by F1 Weighted
    df_res = df_res.sort_values(by="f1_weighted", ascending=False)

    summary_path = EXPERIMENTS_DIR / "leaderboard.csv"
    df_res.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("🏆 TOP 3 CONFIGURATIONS")
    print("=" * 60)
    # Show Threshold Name and Accuracy
    print(
        df_res[["experiment_id", "threshold_mode", "f1_weighted", "accuracy"]]
        .head(3)
        .to_string(index=False)
    )

    best = df_res.iloc[0]

    print("\n" + "=" * 60)
    print("OPTIMAL SETUP FOUND")
    print("=" * 60)
    print(f"Best Threshold Strategy: {best['threshold_mode']}")
    print(f"  Strong Cutoff:  {best['thresh_strong']}%")
    print(f"  Modest Cutoff:  {best['thresh_modest']}%")
    print("-" * 30)
    print("Best Model Parameters:")
    print("params = {")
    print(f"    'n_estimators': {best['n_estimators']},")
    print(f"    'learning_rate': {best['learning_rate']},")
    print(f"    'max_depth': {best['max_depth']},")
    print(f"    'subsample': {best['subsample']},")
    print(f"    'max_features': '{best['max_features']}',")
    print(f"    'random_state': {best['random_state']}")
    print("}")
    print("=" * 60)


if __name__ == "__main__":
    run_experiments()
