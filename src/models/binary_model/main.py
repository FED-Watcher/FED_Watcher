"""
Main script for Fed Market Prediction project.
Orchestrates the complete machine learning pipeline.
"""

import numpy as np
from pathlib import Path
from src.data_preparation import prepare_pipeline
from src.model_training import (
    train_xgboost_model,
    evaluate_model,
    save_model,
    save_metrics,
    get_feature_importance
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


def repeated_time_series_cv(model, X, y, n_splits=4, n_repeats=10, random_state=42):
    """
    Perform repeated time series cross-validation to get more stable estimates.

    Multiple runs help reduce variance in CV scores, especially with small datasets.
    Each repeat uses a different random seed for XGBoost.

    Args:
        model: XGBoost model instance (will be cloned for each repeat)
        X: Feature matrix
        y: Target vector
        n_splits: Number of folds per repeat (default: 4)
        n_repeats: Number of times to repeat CV (default: 10)
        random_state: Base random seed (default: 42)

    Returns:
        dict: Comprehensive CV results with statistics across all repeats
    """
    print("\n🔄 REPEATED TIME SERIES CROSS-VALIDATION")
    print("=" * 70)
    print(f"Configuration: {n_repeats} repeats × {n_splits} folds = {n_repeats * n_splits} total evaluations")

    # Validate inputs
    if len(X) < n_splits + 1:
        print(f"⚠️  Warning: Dataset too small for {n_splits} splits. Using {len(X) - 1} splits.")
        n_splits = max(2, len(X) - 1)

    all_scores = []
    repeat_means = []

    for repeat in range(n_repeats):
        # Clone model with different random seed for each repeat
        model_clone = model.__class__(**model.get_params())
        model_clone.set_params(random_state=random_state + repeat)

        # TimeSeriesSplit for temporal data
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Calculate scores for this repeat
        cv_scores = cross_val_score(
            model_clone, X, y,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )

        all_scores.extend(cv_scores)
        repeat_means.append(cv_scores.mean())

        print(f"Repeat {repeat + 1:2d}/{n_repeats}: Mean = {cv_scores.mean():.4f} ({cv_scores.mean():>6.2%}) | "
              f"Scores = {[f'{s:.3f}' for s in cv_scores]}")

    all_scores = np.array(all_scores)
    repeat_means = np.array(repeat_means)

    # Calculate comprehensive statistics
    print(f"\n{'=' * 70}")
    print("📊 OVERALL RESULTS")
    print(f"{'=' * 70}")
    print(f"\n🎯 Across ALL {len(all_scores)} fold evaluations:")
    print(f"  Overall Mean:     {all_scores.mean():.4f} ({all_scores.mean():.2%})")
    print(f"  Overall Std Dev:  {all_scores.std():.4f}")
    print(f"  Min Score:        {all_scores.min():.4f} ({all_scores.min():.2%})")
    print(f"  Max Score:        {all_scores.max():.4f} ({all_scores.max():.2%})")
    print(f"  Median:           {np.median(all_scores):.4f} ({np.median(all_scores):.2%})")

    print(f"\n📈 Across {n_repeats} repeat means:")
    print(f"  Mean of Means:    {repeat_means.mean():.4f} ({repeat_means.mean():.2%})")
    print(f"  Std of Means:     {repeat_means.std():.4f}")
    print(f"  95% Confidence:   [{repeat_means.mean() - 1.96 * repeat_means.std():.4f}, "
          f"{repeat_means.mean() + 1.96 * repeat_means.std():.4f}]")

    # Stability analysis
    print(f"\n🔬 Stability Analysis:")
    cv_coefficient = (all_scores.std() / all_scores.mean()) * 100
    print(f"  Coefficient of Variation: {cv_coefficient:.2f}%")

    return {
        'all_scores': all_scores.tolist(),
        'repeat_means': repeat_means.tolist(),
        'overall_mean': float(all_scores.mean()),
        'overall_std': float(all_scores.std()),
        'overall_median': float(np.median(all_scores)),
        'mean_of_means': float(repeat_means.mean()),
        'std_of_means': float(repeat_means.std()),
        'min_score': float(all_scores.min()),
        'max_score': float(all_scores.max()),
        'cv_coefficient': float(cv_coefficient),
        'n_repeats': n_repeats,
        'n_splits': n_splits,
        'total_evaluations': len(all_scores),
        'confidence_interval_95': [
            float(repeat_means.mean() - 1.96 * repeat_means.std()),
            float(repeat_means.mean() + 1.96 * repeat_means.std())
        ]
    }


def validate_paths(*paths):
    """Ensure all required directories exist."""
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def main():
    """
    Main execution function for the ML pipeline.
    """
    print("\n" + "=" * 60)
    print("FED MARKET PREDICTION - BINARY CLASSIFICATION")
    print("=" * 60)
    print("Sprint: Predict market direction after Fed announcements")
    print("=" * 60 + "\n")

    # Configuration
    CONFIG = {
        'data_path': 'MasterDataset_Enriched.csv',
        'model_path': 'models/xgboost_binary_classifier.pkl',
        'metrics_path': 'logs/metrics.json',
        'use_announcement_only': True,
        'horizon': 1,
        'test_size': 0.2,
        'cv_splits': 4,
        'cv_repeats': 10  # Number of times to repeat CV
    }

    # Ensure output directories exist
    validate_paths(CONFIG['model_path'], CONFIG['metrics_path'])

    try:
        # ===== STEP 1: DATA PREPARATION =====
        print("📁 Loading and preparing data...")
        X_train, X_test, y_train, y_test, feature_cols = prepare_pipeline(
            filepath=CONFIG['data_path'],
            use_announcement_only=CONFIG['use_announcement_only'],
            horizon=CONFIG['horizon'],
            test_size=CONFIG['test_size']
        )

        # Data validation
        print(f"\n✓ Data prepared successfully")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")

        if len(X_train) < 20:
            print("\n⚠️  Warning: Very small training set (<20 samples)")
            print("    Model may not generalize well")

        # ===== STEP 2: MODEL TRAINING =====
        print("\n🤖 Training XGBoost model...")
        model = train_xgboost_model(X_train, y_train)

        # ===== STEP 3: REPEATED CROSS-VALIDATION =====
        cv_results = repeated_time_series_cv(
            model, X_train, y_train,
            n_splits=CONFIG['cv_splits'],
            n_repeats=CONFIG['cv_repeats']
        )

        # ===== STEP 4: MODEL EVALUATION =====
        print("\n" + "=" * 70)
        print("📊 FINAL MODEL EVALUATION")
        print("=" * 70)
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

        # Add CV results to metrics
        metrics['cross_validation'] = cv_results

        # ===== STEP 5: FEATURE IMPORTANCE =====
        importance_df = get_feature_importance(model, feature_cols)

        # ===== STEP 6: SAVE RESULTS =====
        save_model(model, CONFIG['model_path'])
        save_metrics(metrics, CONFIG['metrics_path'])

        print(f"\n✅ Model saved to: {CONFIG['model_path']}")
        print(f"✅ Metrics saved to: {CONFIG['metrics_path']}")

        # Final summary
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETE!")
        print("=" * 60)
        print("\n📊 Performance Summary:")
        print(f"  {'Metric':<25} {'Score':>10}")
        print(f"  {'-' * 25} {'-' * 10}")
        print(f"  {'Test Accuracy':<25} {metrics['test']['accuracy']:>9.2%}")
        print(f"  {'CV Mean (Overall)':<25} {cv_results['overall_mean']:>9.2%}")
        print(f"  {'CV Std Dev':<25} {cv_results['overall_std']:>9.4f}")
        print(f"  {'CV Median':<25} {cv_results['overall_median']:>9.2%}")
        print(f"  {'Mean of Repeat Means':<25} {cv_results['mean_of_means']:>9.2%}")

        # Show confidence interval
        ci_low, ci_high = cv_results['confidence_interval_95']
        print(f"\n  95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"  ({ci_low:.2%} to {ci_high:.2%})")

        print("=" * 60 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: Data file not found - {e}")
        print("   Please check the file path and try again.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
