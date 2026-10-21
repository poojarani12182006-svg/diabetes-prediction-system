"""
train_model.py
----------------
Trains and compares Logistic Regression, Decision Tree, and Random Forest
classifiers on the Pima Indians Diabetes Dataset, selects the best model
based on F1-score, and saves:
  - model/diabetes_model.joblib   (the trained model)
  - model/scaler.joblib           (the fitted StandardScaler)
  - model/metrics.json            (evaluation metrics for the website)
  - model/confusion_matrix.png    (confusion matrix plot for the best model)
  - model/model_comparison.png    (bar chart comparing all models)

Run this script from the project root:
    python notebooks/train_model.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save PNG files
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

import joblib

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "diabetes.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COLUMN = "Outcome"

# Columns where a value of 0 is biologically impossible and really means
# "missing" in this dataset (Pregnancies and the target are excluded).
ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(df.head())

# ---------------------------------------------------------------------------
# 2. Preprocessing: handle invalid/missing values
# ---------------------------------------------------------------------------
# In this dataset, missing values were originally encoded as 0 for several
# medical measurements where 0 is not physiologically possible. We convert
# those zeros to NaN, then impute them with the median of each column
# (median is more robust to outliers than the mean for skewed medical data).
df[ZERO_AS_MISSING_COLUMNS] = df[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)

print("\nMissing values per column (after converting invalid zeros to NaN):")
print(df.isnull().sum())

for col in ZERO_AS_MISSING_COLUMNS:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

print("\nMissing values after imputation:")
print(df.isnull().sum())

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 4. Feature scaling
# ---------------------------------------------------------------------------
# Logistic Regression benefits strongly from scaled features. Tree-based
# models don't need it, but we scale consistently for all models so that
# the SAME preprocessing pipeline can be reused at prediction time.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 5. Train and compare models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=RANDOM_STATE
    ),
}

results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
    }
    trained_models[name] = model

    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, target_names=["Not Diabetic", "Diabetic"]))
    print("Confusion Matrix:\n", cm)

# ---------------------------------------------------------------------------
# 6. Select the best model (highest F1-score — balances precision & recall,
#    which matters for medical predictions with class imbalance)
# ---------------------------------------------------------------------------
best_model_name = max(results, key=lambda name: results[name]["f1_score"])
best_model = trained_models[best_model_name]
best_metrics = results[best_model_name]

print(f"\n>>> Best model selected: {best_model_name}")
print(f">>> Metrics: {best_metrics}")

# ---------------------------------------------------------------------------
# 7. Save confusion matrix plot for the best model
# ---------------------------------------------------------------------------
cm_best = np.array(best_metrics["confusion_matrix"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=["Not Diabetic", "Diabetic"])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 8. Save model comparison bar chart
# ---------------------------------------------------------------------------
metric_names = ["accuracy", "precision", "recall", "f1_score"]
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(metric_names))
width = 0.25

for i, name in enumerate(results.keys()):
    values = [results[name][m] for m in metric_names]
    ax.bar(x + i * width, values, width, label=name)

ax.set_xticks(x + width)
ax.set_xticklabels([m.replace("_", " ").title() for m in metric_names])
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "model_comparison.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 9. Save trained model, scaler, and metrics to disk
# ---------------------------------------------------------------------------
joblib.dump(best_model, os.path.join(MODEL_DIR, "diabetes_model.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

metrics_output = {
    "best_model": best_model_name,
    "feature_columns": FEATURE_COLUMNS,
    "all_models": results,
}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_output, f, indent=2)

print("\nSaved:")
print(" -", os.path.join(MODEL_DIR, "diabetes_model.joblib"))
print(" -", os.path.join(MODEL_DIR, "scaler.joblib"))
print(" -", os.path.join(MODEL_DIR, "metrics.json"))
print(" -", os.path.join(STATIC_DIR, "confusion_matrix.png"))
print(" -", os.path.join(STATIC_DIR, "model_comparison.png"))
print("\nTraining complete.")
