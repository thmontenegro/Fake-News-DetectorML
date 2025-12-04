#!/usr/bin/env python3
"""
Small I/O utilities shared across training and app code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union, Mapping

PathLike = Union[str, Path]

__all__ = [
    "ensure_outdir",
    "save_json",
    "load_json",
    "compute_classification_metrics",
    "plot_confusion_matrix",
    "plot_curve",
]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def ensure_outdir(path: PathLike) -> Path:
    """
    Ensure a directory exists; create parents if needed.

    Parameters
    ----------
    path : str | Path
        Directory path to create/ensure.

    Returns
    -------
    Path
        Resolved directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def save_json(obj: Mapping[str, Any] | Any, path: PathLike, *, indent: int = 2) -> Path:
    """
    Save a Python object as pretty-printed JSON.

    Parameters
    ----------
    obj : Any
        JSON-serializable object.
    path : str | Path
        Output file path.
    indent : int
        JSON indent spacing.

    Returns
    -------
    Path
        The written file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    return p.resolve()


def load_json(path: PathLike) -> Any:
    """
    Load JSON from disk.

    Parameters
    ----------
    path : str | Path
        JSON file path.

    Returns
    -------
    Any
        Parsed JSON.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_classification_metrics(y_true, y_pred, y_prob=None, *, target_names=None):
    """
    Compute a standard set of classification metrics and return as a dict.

    Parameters
    ----------
    y_true : array-like
        True labels (0/1 or similar).
    y_pred : array-like
        Predicted labels.
    y_prob : array-like or None
        Predicted probability for the positive class (optional).
    target_names : list[str] or None
        Optional names for classes used in classification report.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision/recall/F1 (macro and weighted),
        classification report (as dict), confusion matrix (2D list), and optional ROC/AP.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    out = {
        "accuracy": acc,
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    if y_prob is not None:
        try:
            roc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc = None
        try:
            ap = float(average_precision_score(y_true, y_prob))
        except Exception:
            ap = None
        out.update({"roc_auc": roc, "avg_precision": ap})

    return out


def plot_confusion_matrix(cm: np.ndarray, out: PathLike, labels=None, title: str = "Confusion Matrix") -> Path:
    """Plot and save a confusion matrix image.

    Returns the written Path.
    """
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if labels is None:
        labels = ["0", "1"]

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(outp, dpi=150)
    plt.close(fig)
    return outp.resolve()


def plot_curve(x, y, out: PathLike, title: str, xlabel: str, ylabel: str) -> Path:
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(outp, dpi=150)
    plt.close(fig)
    return outp.resolve()
