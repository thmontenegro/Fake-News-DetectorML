#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from utils import (
    ensure_outdir,
    save_json,
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_curve,
)

# -----------------------------
# Helpers
# -----------------------------
LABELS: Final[Tuple[str, str]] = ("REAL", "FAKE")


def read_csv_any(path: Path, nrows: int | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=nrows, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=nrows, encoding="latin-1")


# -----------------------------
# Plot utilities
# -----------------------------
# plotting and metric helpers are provided by `src/utils.py`


# -----------------------------
# Main training routine
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train Fake News Detector with TF-IDF (1–3 grams) + RandomForest"
    )
    ap.add_argument("--real", required=True, help="Path to True.csv")
    ap.add_argument("--fake", required=True, help="Path to Fake.csv")
    ap.add_argument(
        "--text-col",
        default="text",
        help="Name of the text column (before combining). Default: text",
    )
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    a = ap.parse_args()

    outdir = ensure_outdir(Path(a.outdir))
    charts = ensure_outdir(outdir / "charts")

    # 1) Load data
    df_real = read_csv_any(Path(a.real))
    df_fake = read_csv_any(Path(a.fake))

    # 2) Build 'combined_text' = title + text (more signal)
    for df in (df_real, df_fake):
        title = df["title"].fillna("") if "title" in df.columns else ""
        txt = df.get(a.text_col, df.get("text", "")).fillna("")
        df["combined_text"] = (title + " " + txt).str.strip()

    # 3) Prepare X, y with balanced labels
    X = pd.concat([df_real["combined_text"], df_fake["combined_text"]], ignore_index=True)
    y = np.array([0] * len(df_real) + [1] * len(df_fake))  # 0=REAL, 1=FAKE

    # 4) Train/Validation split (stratified hold-out)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 5) Pipeline: TF-IDF (1–3 grams) + RandomForest
    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    sublinear_tf=True,
                    stop_words="english",
                    ngram_range=(1, 3),
                    max_df=0.8,
                    min_df=3,
                    max_features=20_000,
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    random_state=42,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 6) 5-fold CV on training split for robust estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"Cross-validated F1 (train split, 5-fold): {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    # 7) Fit on full training split
    pipe.fit(X_train, y_train)

    # 8) Evaluate on hold-out test split
    # 8) Evaluate on hold-out test split
    if hasattr(pipe, "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]
    else:
        # fallback: use decision_function if available
        if hasattr(pipe, "decision_function"):
            raw = pipe.decision_function(X_test)
            # try to map to [0,1]
            y_prob = (raw - raw.min()) / (raw.max() - raw.min())
        else:
            y_prob = None

    y_pred = pipe.predict(X_test)

    metrics = compute_classification_metrics(y_test, y_pred, y_prob=y_prob, target_names=LABELS)
    metrics.update({"cv_f1_macro_mean": float(cv_f1.mean()), "cv_f1_macro_std": float(cv_f1.std())})

    # 9) Save metrics and figures
    save_json(metrics, outdir / "metrics.json")

    # confusion matrix image
    cm = np.array(metrics["confusion_matrix"]) if isinstance(metrics.get("confusion_matrix"), list) else confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(np.asarray(cm), charts / "confusion_matrix.png", labels=LABELS)

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plot_curve(fpr, tpr, charts / "roc_curve.png", "ROC Curve", "FPR", "TPR")

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        plot_curve(rec, prec, charts / "pr_curve.png", "Precision-Recall Curve", "Recall", "Precision")

    # 10) Persist artifacts
    #   Save the whole pipeline as 'model.joblib' (vectorizer+model together)
    joblib.dump(pipe, outdir / "pipeline.joblib")
    #   Also keep separate parts for compatibility with your existing CLI
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    joblib.dump(vec, outdir / "vectorizer.joblib")
    joblib.dump(clf, outdir / "model.joblib")

    print("Training complete. Artifacts saved to:", str(outdir.resolve()))
    print("Key metrics:", json.dumps({k: metrics[k] for k in ['accuracy','roc_auc','avg_precision']}, indent=2))
    print("CV F1 (macro):", f"{metrics['cv_f1_macro_mean']:.3f} ± {metrics['cv_f1_macro_std']:.3f}")
    print("Tip: If you want higher FAKE recall, use a decision threshold < 0.5 when classifying.")
    

if __name__ == "__main__":
    main()
