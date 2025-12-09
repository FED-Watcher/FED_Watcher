"""
FED Watcher - ENHANCED Multi-Class Classification Model
=======================================================

This script implements an ENHANCED multi-class classification model using
the enriched dataset with macroeconomic features (VIX, DXY, yield curve, etc.)

Author: ML Engineering Team
Project: FED Watcher - Enhanced Version
"""

from sklearn.utils.class_weight import compute_sample_weight
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import os
import json

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

os.makedirs("./outputs", exist_ok=True)

print("=" * 70)
print("FED WATCHER - ENHANCED MULTI-CLASS CLASSIFICATION MODEL")
print("Using Enriched Dataset with Macroeconomic Features")
print("=" * 70)
print()

# ============================================================================
# STEP 1: DATA INGESTION & PREPARATION
# ============================================================================
print("STEP 1: Loading Enriched Dataset")
print("-" * 70)

# Load the enriched dataset
# Use Path to navigate to project root data directory

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
df = pd.read_csv(PROJECT_ROOT / "data" / "MasterDataset_Enriched.csv")
print(f"✓ Enriched dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Identify new features
original_features = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Volume_ratio_vs_5days",
    "Announcement",
    "hawkish_dovish_ratio",
    "net_sentiment_score",
    "negative_proportion",
    "neutral_proportion",
    "positive_proportion",
    "sentiment_label",
]
new_features = [col for col in df.columns if col not in original_features]
print(f"✓ New macroeconomic features: {', '.join(new_features)}")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df = df.sort_values("Date").reset_index(drop=True)
print(f"✓ Data sorted chronologically from {df['Date'].min()} to {df['Date'].max()}")

# Filter for announcement days only
df_announcements = df[df["Announcement"] == 1].copy()
print(f"✓ Filtered to announcement days: {len(df_announcements)} events")
print()

# ============================================================================
# STEP 2: TARGET VARIABLE ENGINEERING
# ============================================================================
print("STEP 2: Engineering Multi-Class Target Variable")
print("-" * 70)

# Calculate 24-hour forward returns
df_announcements["Next_Close"] = df_announcements["Close"].shift(-1)
df_announcements["market_return"] = (
    (df_announcements["Next_Close"] - df_announcements["Close"]) / df_announcements["Close"] * 100
)

# Remove the last row
df_announcements = df_announcements[:-1].copy()
print(f"✓ Calculated 24-hour forward returns for {len(df_announcements)} events")

# Use the same thresholds as before for consistency
THRESHOLDS = {
    "strong_rise": 7.5,
    "modest_rise": 3.75,
    "neutral_upper": 3.75,
    "neutral_lower": -3.75,
    "modest_drop": -3.75,
    "strong_drop": -7.5,
}


def classify_return(return_pct):
    if return_pct > THRESHOLDS["strong_rise"]:
        return 2
    elif return_pct > THRESHOLDS["modest_rise"]:
        return 1
    elif return_pct >= THRESHOLDS["neutral_lower"]:
        return 0
    elif return_pct >= THRESHOLDS["strong_drop"]:
        return -1
    else:
        return -2


df_announcements["market_magnitude_class"] = df_announcements["market_return"].apply(
    classify_return
)

# Check class distribution
print("\n--- Class Distribution ---")
class_dist = df_announcements["market_magnitude_class"].value_counts().sort_index()
print(class_dist)
print()

# ============================================================================
# STEP 3: ENHANCED FEATURE SELECTION
# ============================================================================
print("STEP 3: Enhanced Feature Selection")
print("-" * 70)

# Original sentiment features
SENTIMENT_FEATURES = [
    "hawkish_dovish_ratio",
    "net_sentiment_score",
    "negative_proportion",
    "neutral_proportion",
    "positive_proportion",
]

# New macroeconomic features
MACRO_FEATURES = ["VIX_Close", "DXY_Close", "US02Y_Yield", "US10Y_Yield", "Yield_Curve_10Y_2Y"]

# Contextual features
CONTEXTUAL_FEATURES = ["Volume_ratio_vs_5days"]

# Combine all features
FEATURES = SENTIMENT_FEATURES + MACRO_FEATURES + CONTEXTUAL_FEATURES

print("Feature Set Composition:")
print(f"\n1. Sentiment Features ({len(SENTIMENT_FEATURES)}):")
for i, feat in enumerate(SENTIMENT_FEATURES, 1):
    print(f"   {i}. {feat}")

print(f"\n2. Macroeconomic Features ({len(MACRO_FEATURES)}) - NEW:")
for i, feat in enumerate(MACRO_FEATURES, 1):
    print(f"   {i}. {feat}")

print(f"\n3. Contextual Features ({len(CONTEXTUAL_FEATURES)}):")
for i, feat in enumerate(CONTEXTUAL_FEATURES, 1):
    print(f"   {i}. {feat}")

print(f"\nTotal Features: {len(FEATURES)}")
print()

# Check for missing values
missing_check = df_announcements[FEATURES + ["market_magnitude_class"]].isnull().sum()
if missing_check.sum() > 0:
    print("⚠ Missing values detected:")
    print(missing_check[missing_check > 0])
    print("\nRemoving rows with missing values...")
    df_announcements = df_announcements.dropna(subset=FEATURES + ["market_magnitude_class"])
    print(f"✓ Dataset size after removing missing values: {len(df_announcements)} events")
else:
    print("✓ No missing values in feature set")
print()

# ============================================================================
# STEP 4: FEATURE ANALYSIS & CORRELATION
# ============================================================================
print("STEP 4: Analyzing New Features")
print("-" * 70)

# Create correlation matrix for new features
feature_data = df_announcements[FEATURES].copy()
correlation_matrix = feature_data.corr()

# Plot correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    ax=ax,
    cbar_kws={"label": "Correlation"},
)
ax.set_title("Feature Correlation Matrix (Enhanced Feature Set)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("./outputs/enhanced_feature_correlation.png", dpi=300, bbox_inches="tight")
print("✓ Feature correlation analysis saved")

# Analyze macro features on announcement days
print("\nMacroeconomic Feature Statistics (Announcement Days):")
print(df_announcements[MACRO_FEATURES].describe().round(2))
print()

# ============================================================================
# STEP 5: CHRONOLOGICAL DATA SPLITTING WITH SCALING
# ============================================================================
print("STEP 5: Chronological Train-Test Split with Feature Scaling")
print("-" * 70)

# Sort by date
df_announcements = df_announcements.sort_values("Date").reset_index(drop=True)

# Calculate split point (70-30 split)
split_index = int(len(df_announcements) * 0.7)

train_data = df_announcements.iloc[:split_index].copy()
test_data = df_announcements.iloc[split_index:].copy()

print("Training Set:")
print(f"  Size: {len(train_data)} events")
print(f"  Date Range: {train_data['Date'].min()} to {train_data['Date'].max()}")

print("\nTest Set:")
print(f"  Size: {len(test_data)} events")
print(f"  Date Range: {test_data['Date'].min()} to {test_data['Date'].max()}")
print()

# Separate features and target
X_train_raw = train_data[FEATURES].values
y_train = train_data["market_magnitude_class"].values

X_test_raw = test_data[FEATURES].values
y_test = test_data["market_magnitude_class"].values

# Apply feature scaling (important for features with different scales)
print("Applying StandardScaler to normalize features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
print("✓ Feature scaling complete")
print()

print("Feature matrices created:")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")
print()

# ============================================================================
# STEP 6: ENHANCED MODEL TRAINING
# ============================================================================
print("STEP 6: Training Enhanced Multi-Class Model")
print("-" * 70)

# Calculate class weights
sample_weights = compute_sample_weight("balanced", y_train)

print("Model Configuration:")
print("  Algorithm: Gradient Boosting Classifier")
print("  Enhanced Features: Sentiment + Macroeconomic + Volume")
print("  Feature Scaling: StandardScaler applied")
print("  Class Weighting: Balanced")
print()

# FED-18.2: Optimized hyperparameters for improved prediction accuracy
# Based on grid search results from cross-validation experiments
model_params = {
    "n_estimators": 200,  # Optimized
    "max_depth": 5,  # Optimized
    "learning_rate": 0.05,  # Optimized (Slower learning)
    "subsample": 0.8,  # Optimized
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "verbose": 0,
}

print("Hyperparameters (Enhanced for Accuracy):")
for key, value in model_params.items():
    print(f"  {key}: {value}")
print()

# Remap classes
class_mapping = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
inverse_mapping = {v: k for k, v in class_mapping.items()}

y_train_mapped = np.array([class_mapping[y] for y in y_train])
y_test_mapped = np.array([class_mapping[y] for y in y_test])

# Train the model
print("Training enhanced model...")
model = GradientBoostingClassifier(**model_params)
model.fit(X_train, y_train_mapped, sample_weight=sample_weights)
print("✓ Enhanced model training complete!")
print()

# ============================================================================
# STEP 7: MODEL EVALUATION
# ============================================================================
print("STEP 7: Evaluating Enhanced Model")
print("-" * 70)

# Generate predictions
y_pred_mapped = model.predict(X_test)
y_pred = np.array([inverse_mapping[y] for y in y_pred_mapped])

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print()

# Generate confusion matrix
print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred, labels=[-2, -1, 0, 1, 2])
print(cm)
print()

# Visualize confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
class_labels = [
    "Strong Drop/n(-2)",
    "Modest Drop/n(-1)",
    "Neutral/n(0)",
    "Modest Rise/n(1)",
    "Strong Rise/n(2)",
]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap="Blues", ax=ax, values_format="d")
ax.set_title("Confusion Matrix - Enhanced Multi-Class Model", fontsize=14, fontweight="bold")
plt.xticks(rotation=0, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("./outputs/enhanced_confusion_matrix.png", dpi=300, bbox_inches="tight")
print("✓ Enhanced confusion matrix saved")
print()

# Classification report
print("--- Classification Report ---")
report = classification_report(
    y_test, y_pred, labels=[-2, -1, 0, 1, 2], target_names=class_labels, digits=4
)
print(report)

# Save report
with open("./outputs/enhanced_classification_report.txt", "w") as f:
    f.write("Enhanced Multi-Class Classification Model - Evaluation Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Features Used: {len(FEATURES)}/n")
    f.write(f"  - Sentiment: {len(SENTIMENT_FEATURES)}\n")
    f.write(f"  - Macroeconomic: {len(MACRO_FEATURES)}\n")
    f.write(f"  - Contextual: {len(CONTEXTUAL_FEATURES)}\n\n")
    f.write(f"Test Set Size: {len(y_test)} events\n")
    f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Per-Class Performance:\n")
    f.write(report)

print("✓ Enhanced classification report saved")
print()

# ============================================================================
# STEP 8: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("STEP 8: Enhanced Feature Importance Analysis")
print("-" * 70)

# Get feature importances
feature_importance_scores = model.feature_importances_

# Create dataframe
importance_df = pd.DataFrame(
    {
        "Feature": FEATURES,
        "Importance": feature_importance_scores,
        "Category": ["Sentiment"] * len(SENTIMENT_FEATURES)
        + ["Macro"] * len(MACRO_FEATURES)
        + ["Context"] * len(CONTEXTUAL_FEATURES),
    }
).sort_values("Importance", ascending=False)

# Plot feature importance with categories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Individual features
importance_df_sorted = importance_df.sort_values("Importance", ascending=True)
colors = [
    "steelblue" if cat == "Sentiment" else "darkgreen" if cat == "Macro" else "coral"
    for cat in importance_df_sorted["Category"]
]
bars = ax1.barh(
    importance_df_sorted["Feature"], importance_df_sorted["Importance"], color=colors, alpha=0.8
)
ax1.set_xlabel("Feature Importance", fontsize=12, fontweight="bold")
ax1.set_title("Feature Importance - Enhanced Model", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3, axis="x")

# Add legend
legend_elements = [
    Patch(facecolor="steelblue", alpha=0.8, label="Sentiment"),
    Patch(facecolor="darkgreen", alpha=0.8, label="Macroeconomic"),
    Patch(facecolor="coral", alpha=0.8, label="Contextual"),
]
ax1.legend(handles=legend_elements, loc="lower right")

# Category-level importance
category_importance = importance_df.groupby("Category")["Importance"].sum().sort_values()
ax2.barh(
    category_importance.index,
    category_importance.values,
    color=["coral", "darkgreen", "steelblue"],
    alpha=0.8,
)
ax2.set_xlabel("Total Importance", fontsize=12, fontweight="bold")
ax2.set_title("Feature Category Importance", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3, axis="x")

# Add percentage labels
for i, (cat, val) in enumerate(category_importance.items()):
    pct = val / category_importance.sum() * 100
    ax2.text(val, i, f" {pct:.1f}%", va="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("./outputs/enhanced_feature_importance.png", dpi=300, bbox_inches="tight")
print("✓ Enhanced feature importance plot saved")
print()

print("Top 10 Most Important Features:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {row['Feature']:30s} ({row['Category']:10s}): {row['Importance']:.6f}")
print()

print("Feature Category Contributions:")
for cat, val in category_importance.sort_values(ascending=False).items():
    pct = val / category_importance.sum() * 100
    print(f"  {cat:15s}: {val:.4f} ({pct:.1f}%)")
print()

# ============================================================================
# STEP 9: SAVE ENHANCED MODEL
# ============================================================================
print("STEP 9: Saving Enhanced Model and Artifacts")
print("-" * 70)

# Save model
model_filename = "./outputs/enhanced_multi_class_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(model, f)
print("✓ Enhanced model saved")

# Save scaler
scaler_filename = "./outputs/feature_scaler.pkl"
with open(scaler_filename, "wb") as f:
    pickle.dump(scaler, f)
print("✓ Feature scaler saved")

# Save metadata
metadata = {
    "model_type": "Enhanced Gradient Boosting Multi-Class Classifier",
    "features": FEATURES,
    "feature_categories": {
        "sentiment": SENTIMENT_FEATURES,
        "macroeconomic": MACRO_FEATURES,
        "contextual": CONTEXTUAL_FEATURES,
    },
    "class_mapping": class_mapping,
    "inverse_mapping": inverse_mapping,
    "thresholds": THRESHOLDS,
    "train_size": len(train_data),
    "test_size": len(test_data),
    "train_date_range": f"{train_data['Date'].min()} to {train_data['Date'].max()}",
    "test_date_range": f"{test_data['Date'].min()} to {test_data['Date'].max()}",
    "hyperparameters": model_params,
    "test_accuracy": float(accuracy),
    "feature_scaling": "StandardScaler",
}

with open("./outputs/enhanced_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4, default=str)
print("✓ Enhanced model metadata saved")
print()

# ============================================================================
# STEP 10: SAVE PREDICTIONS
# ============================================================================
print("STEP 10: Saving Enhanced Model Predictions")
print("-" * 70)

# Get prediction probabilities
y_pred_proba = model.predict_proba(X_test)

# Create predictions dataframe
predictions_df = pd.DataFrame(
    {
        "meeting_date": test_data["Date"].values,
        "actual_return_pct": test_data["market_return"].values,
        "actual_class": y_test,
        "predicted_class": y_pred,
        "prob_strong_drop": y_pred_proba[:, 0],
        "prob_modest_drop": y_pred_proba[:, 1],
        "prob_neutral": y_pred_proba[:, 2],
        "prob_modest_rise": y_pred_proba[:, 3],
        "prob_strong_rise": y_pred_proba[:, 4],
    }
)

# Add labels
class_label_map = {
    -2: "Strong Drop",
    -1: "Modest Drop",
    0: "Neutral",
    1: "Modest Rise",
    2: "Strong Rise",
}

predictions_df["actual_class_label"] = predictions_df["actual_class"].map(class_label_map)
predictions_df["predicted_class_label"] = predictions_df["predicted_class"].map(class_label_map)
predictions_df["correct_prediction"] = (
    predictions_df["actual_class"] == predictions_df["predicted_class"]
)

# Save
predictions_filename = "./outputs/enhanced_test_predictions.csv"
predictions_df.to_csv(predictions_filename, index=False)
print("✓ Enhanced predictions saved")
print()

# ============================================================================
# STEP 11: COMPARISON WITH BASELINE
# ============================================================================
print("STEP 11: Comparison with Baseline Model")
print("-" * 70)

# Load baseline predictions if available
try:
    baseline_preds = pd.read_csv("./outputs/test_predictions_multiclass.csv")
    baseline_accuracy = (baseline_preds["actual_class"] == baseline_preds["predicted_class"]).mean()

    print("Performance Comparison:")
    print(f"  Baseline Model Accuracy:  {baseline_accuracy:.4f} ({baseline_accuracy * 100:.2f}%)")
    print(f"  Enhanced Model Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")

    improvement = ((accuracy - baseline_accuracy) / baseline_accuracy) * 100
    if improvement > 0:
        print(f"  Improvement:              +{improvement:.2f}%")
        print(
            f"  Absolute Gain:            +{(accuracy - baseline_accuracy) * 100:.2f} percentage points"
        )
    else:
        print(f"  Change:                   {improvement:.2f}%")

    print()

    # Create comparison visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ["Baseline/n(Sentiment Only)", "Enhanced/n(+ Macro Features)"]
    accuracies = [baseline_accuracy, accuracy]
    colors = ["steelblue", "darkgreen"]

    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor="black", linewidth=2)
    ax.set_ylabel("Test Set Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% (Random Baseline)")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.2%}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.legend()
    plt.tight_layout()
    plt.savefig("./outputs/model_comparison.png", dpi=300, bbox_inches="tight")
    print("✓ Model comparison visualization saved")

except FileNotFoundError:
    print("⚠ Baseline predictions not found - skipping comparison")

print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 70)
print("ENHANCED MODEL TRAINING COMPLETE")
print("=" * 70)
print()
print("✓ Enhanced model with macroeconomic features successfully trained!")
print()
print("Key Improvements:")
print(f"  • Total Features: {len(FEATURES)} (vs 6 in baseline)")
print("  • New Macro Features: VIX, DXY, yields, yield curve")
print("  • Feature Scaling: StandardScaler applied")
print("  • Model Capacity: Increased (150 estimators, depth 5)")
print()
print("Deliverables:")
print("  1. enhanced_multi_class_model.pkl")
print("  2. feature_scaler.pkl (NEW - required for predictions)")
print("  3. enhanced_model_metadata.json")
print("  4. enhanced_confusion_matrix.png")
print("  5. enhanced_classification_report.txt")
print("  6. enhanced_feature_importance.png")
print("  7. enhanced_feature_correlation.png (NEW)")
print("  8. enhanced_test_predictions.csv")
print("  9. model_comparison.png (if baseline available)")
print()
print("Enhanced Model Performance:")
print(f"  Test Accuracy: {accuracy * 100:.2f}%")
print(f"  Test Set Size: {len(y_test)} events")
print()
print("=" * 70)
