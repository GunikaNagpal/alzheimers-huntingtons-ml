# scripts/train_model.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, accuracy_score
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

# --------- paths ----------
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
labels = pd.read_csv(OUT / "labels.csv")                 # SampleID, Diagnosis
expr   = pd.read_csv(OUT / "expression.csv", index_col=0) # genes x samples

# --------- align ----------
X = expr.T.replace([np.inf, -np.inf], np.nan)
X = X.loc[labels["SampleID"]].dropna(axis=1)  # keep genes with no NaNs
y = labels["Diagnosis"].values
classes = sorted(pd.unique(y))

print(f"Aligned matrix: {X.shape[0]} samples x {X.shape[1]} genes")
print("Class counts:\n", pd.Series(y).value_counts())

# --------- split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

# --------- feature selection + scaling ----------
# Keep only informative genes:
# 1) remove near-constant features; 2) keep top-K by ANOVA F-score
K_TOP = 2000  # tweakable
pipe_features = Pipeline([
    ("var", VarianceThreshold(threshold=0.0)),
    ("scale", StandardScaler(with_mean=True, with_std=True)),
    ("kbest", SelectKBest(score_func=f_classif, k=min(K_TOP, X.shape[1]))),
])

X_train_fs = pipe_features.fit_transform(X_train, y_train)
X_test_fs  = pipe_features.transform(X_test)

print("After FS:", X_train_fs.shape, "(train)")

# --------- models to compare ----------
models = {
    "LogReg-OVR":  LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="ovr", n_jobs=None),
    "SVM-RBF":     SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, class_weight="balanced"),
    "RandomForest":RandomForestClassifier(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1, class_weight="balanced")
}

# --------- CV comparison (on train set) ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, model in models.items():
    # small pipeline so CV includes the model only (features are precomputed above)
    scores = cross_val_score(model, X_train_fs, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_results[name] = (scores.mean(), scores.std())
    print(f"[CV] {name}: {scores.mean():.3f} ± {scores.std():.3f}")

# choose best by mean CV
best_name = max(cv_results, key=lambda k: cv_results[k][0])
best_model = models[best_name]
print(f"\nSelected model: {best_name}")

# --------- fit best model ----------
best_model.fit(X_train_fs, y_train)
y_pred = best_model.predict(X_test_fs)
report = classification_report(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print("\nTest accuracy:", acc)
print("\nClassification report:\n", report)

(Path(OUT / "classification_report.txt")).write_text(
    f"Model: {best_name}\nAccuracy: {acc:.4f}\n\n{report}"
)
print("Saved:", OUT / "classification_report.txt")

# --------- confusion matrix (matplotlib only) ----------
cm = confusion_matrix(y_test, y_pred, labels=classes)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation="nearest")
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(ticks=np.arange(len(classes)), labels=classes, rotation=45)
plt.yticks(ticks=np.arange(len(classes)), labels=classes)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=150)
plt.close()

# --------- PCA plot (2D) ----------
from sklearn.decomposition import PCA
X_full_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(X.values)  # for visualization only
pca = PCA(n_components=2, random_state=42).fit_transform(X_full_scaled)
label_series = pd.Series(labels["Diagnosis"].values, index=labels["SampleID"].values)

plt.figure()
for cls in classes:
    mask = (label_series.values == cls)
    plt.scatter(pca[mask, 0], pca[mask, 1], s=12, label=cls)
plt.title("PCA (2D) — All Samples")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "pca_scatter.png", dpi=150)
plt.close()

# --------- Heatmap of top-variance genes ----------
# (pure matplotlib; index labels omitted to keep it readable)
variances = X.var(axis=0).sort_values(ascending=False)
N = 50
top_genes = variances.head(N).index
X_top = X[top_genes]
Z = (X_top - X_top.mean()) / X_top.std(ddof=0)
plt.figure(figsize=(10,8))
plt.imshow(Z.values, aspect="auto")
plt.title(f"Heatmap: Top {N} Most Variable Genes")
plt.xlabel("Genes"); plt.ylabel("Samples")
plt.tight_layout()
plt.savefig(OUT / "heatmap_top_variance.png", dpi=150)
plt.close()

# --------- ROC (one-vs-rest) ----------
# Need predict_proba; if SVM without probability, we enabled probability=True above
y_test_bin = label_binarize(y_test, classes=classes)
if hasattr(best_model, "predict_proba"):
    y_score = best_model.predict_proba(X_test_fs)
else:
    # fallback: calibrate via decision_function if available
    if hasattr(best_model, "decision_function"):
        scores = best_model.decision_function(X_test_fs)
        # decision_function returns shape (n_samples, n_classes)
        # convert to [0,1] via min-max per class (quick-n-dirty)
        scores = (scores - scores.min(0)) / (scores.max(0) - scores.min(0) + 1e-9)
        y_score = scores
    else:
        y_score = None

if y_score is not None:
    plt.figure()
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.2f})")
    plt.plot([0,1],[0,1], "--")
    plt.title(f"ROC (OvR) — {best_name}")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "roc_curves.png", dpi=150)
    plt.close()

# --------- Feature importance (if model supports it) ----------
if hasattr(best_model, "feature_importances_"):
    # Map importances back to original gene names after SelectKBest
    # Recover selected feature indices
    kbest = pipe_features.named_steps["kbest"]
    selected_idx = kbest.get_support(indices=True)
    # names after var+scale+kbest correspond to the subset of original columns
    original_cols = np.array(X.columns)
    selected_cols = original_cols[selected_idx]

    importances = best_model.feature_importances_
    order = np.argsort(importances)[::-1][:20]
    top_names = selected_cols[order]
    top_scores = importances[order]
    pd.DataFrame({"gene": top_names, "importance": top_scores}).to_csv(OUT/"top20_feature_importance.csv", index=False)

    plt.figure(figsize=(8,6))
    pos = np.arange(len(top_names))
    plt.bar(pos, top_scores)
    plt.xticks(pos, top_names, rotation=90, fontsize=6)
    plt.title("Top 20 Features (RandomForest)")
    plt.xlabel("Genes"); plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance_top20.png", dpi=150)
    plt.close()

print("\n✅ Done. Check the 'outputs/' folder for:")
print("- classification_report.txt")
print("- confusion_matrix.png")
print("- pca_scatter.png")
print("- heatmap_top_variance.png")
print("- roc_curves.png (if available)")
print("- top20_feature_importance.csv / feature_importance_top20.png (if RF selected)")
