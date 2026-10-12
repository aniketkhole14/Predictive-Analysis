"""
Breast Cancer Diagnosis Prediction using Binomial Logistic Regression
Course: Predictive Analytics

Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository
         (bundled with scikit-learn, so no download is needed)

Pipeline:
  1. Load data & basic EDA
  2. Train/test split (stratified)
  3. Standardise features + Logistic Regression (L2) with GridSearchCV on C
  4. Evaluate: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
  5. Interpret: odds ratios of the most influential features
  6. Save plots, metrics and the trained model to ./outputs
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ 1. DATA
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
# In sklearn: target 0 = malignant, 1 = benign. We flip so 1 = malignant
# (the "event" of interest), which makes odds ratios easier to read.
df["target"] = 1 - df["target"]
X = df.drop(columns="target")
y = df["target"]

print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Class balance (1 = malignant):")
print(y.value_counts().rename({0: "benign", 1: "malignant"}))

df.describe().T.to_csv(f"{OUT}/summary_statistics.csv")

# ------------------------------------------------------------------ 2. EDA
plt.figure(figsize=(5, 4))
sns.countplot(x=y.map({0: "Benign", 1: "Malignant"}), palette=["#4C9F70", "#D1495B"])
plt.title("Class distribution")
plt.tight_layout()
plt.savefig(f"{OUT}/01_class_distribution.png", dpi=150)
plt.close()

mean_cols = [c for c in X.columns if c.startswith("mean")]
plt.figure(figsize=(11, 9))
sns.heatmap(X[mean_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.7})
plt.title("Correlation of mean features")
plt.tight_layout()
plt.savefig(f"{OUT}/02_correlation_heatmap.png", dpi=150)
plt.close()

top_vis = ["mean radius", "mean concavity", "mean concave points", "mean texture"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, top_vis):
    sns.boxplot(x=y.map({0: "Benign", 1: "Malignant"}), y=X[col], palette=["#4C9F70", "#D1495B"], ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/03_feature_boxplots.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 3. MODEL
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
])
param_grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("\nBest C:", grid.best_params_["clf__C"], "| CV ROC-AUC:", round(grid.best_score_, 4))

cv_acc = cross_val_score(best, X_train, y_train, cv=cv, scoring="accuracy")
print("5-fold CV accuracy: %.4f +/- %.4f" % (cv_acc.mean(), cv_acc.std()))

# ------------------------------------------------------------------ 4. EVALUATION
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:, 1]

metrics = {
    "best_C": grid.best_params_["clf__C"],
    "cv_accuracy_mean": float(cv_acc.mean()),
    "cv_accuracy_std": float(cv_acc.std()),
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "test_precision": float(precision_score(y_test, y_pred)),
    "test_recall": float(recall_score(y_test, y_pred)),
    "test_f1": float(f1_score(y_test, y_pred)),
    "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nTest metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
print("\n", classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(cmap="Blues", colorbar=False)
plt.title("Confusion matrix (test set)")
plt.tight_layout()
plt.savefig(f"{OUT}/04_confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(5.5, 5))
plt.plot(fpr, tpr, lw=2, label=f"Logistic Regression (AUC = {metrics['test_roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/05_roc_curve.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 5. INTERPRETATION
coefs = best.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": coefs,
    # per +1 standard deviation increase in the feature
    "odds_ratio_per_SD": np.exp(coefs),
}).sort_values("coefficient", key=np.abs, ascending=False)
coef_df.to_csv(f"{OUT}/coefficients_odds_ratios.csv", index=False)

top = coef_df.head(12).iloc[::-1]
plt.figure(figsize=(8, 6))
plt.barh(top["feature"], top["coefficient"],
         color=["#D1495B" if c > 0 else "#4C9F70" for c in top["coefficient"]])
plt.axvline(0, color="k", lw=0.8)
plt.xlabel("Standardised coefficient (log-odds of malignant)")
plt.title("Top 12 features driving the prediction")
plt.tight_layout()
plt.savefig(f"{OUT}/06_feature_importance.png", dpi=150)
plt.close()

print("Top 5 features by |coefficient|:")
print(coef_df.head(5).to_string(index=False))

# ------------------------------------------------------------------ 6. SAVE MODEL
joblib.dump(best, f"{OUT}/logistic_regression_model.joblib")
print(f"\nAll outputs saved in ./{OUT}/")
