"""
Evaluation metrics for LSD600 Multi-Label Relation Extraction.

Implements:
1. Micro- and Macro-averaged Precision, Recall, and F1-score across all 8 relation classes.
2. Per-class Precision, Recall, F1, and support.
3. Coarse-grained evaluation (Positive vs Negative vs Association vs None) matching the paper.
4. Confusion matrix and error analysis summaries.
"""

from typing import Dict, List, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
from src.data.preprocessing import RELATION_CLASSES, RELATION2ID, ID2RELATION


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = RELATION_CLASSES,
) -> Dict[str, Any]:
    """
    Computes standard multi-label relation extraction metrics.

    Args:
        y_true: Binary matrix of shape (N, num_classes)
        y_pred: Binary matrix of shape (N, num_classes)
        class_names: List of class name strings

    Returns:
        dict containing micro, macro, and per-class precision, recall, f1, support.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Micro average
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    # Macro average
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Per class metrics
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class = {}
    for idx, cname in enumerate(class_names):
        per_class[cname] = {
            "precision": float(p_per[idx]),
            "recall": float(r_per[idx]),
            "f1": float(f1_per[idx]),
            "support": int(sup_per[idx]),
            "predicted_count": int(np.sum(y_pred[:, idx])),
        }

    return {
        "micro": {
            "precision": float(p_micro),
            "recall": float(r_micro),
            "f1": float(f1_micro),
        },
        "macro": {
            "precision": float(p_macro),
            "recall": float(r_macro),
            "f1": float(f1_macro),
        },
        "per_class": per_class,
    }


def map_to_coarse_grained(y_binary: np.ndarray) -> np.ndarray:
    """
    Collapses the 8 fine-grained relation classes into coarse categories:
      0: Positive (causes, positive_statistical_association)
      1: Negative (negative_statistical_association, controls, prevents, treats)
      2: Statistical Association (statistical_association)
      3: No Association (no_statistical_association)

    Returns:
        coarse binary matrix of shape (N, 4)
    """
    pos_idx = [RELATION2ID["causes"], RELATION2ID["positive_statistical_association"]]
    neg_idx = [
        RELATION2ID["negative_statistical_association"],
        RELATION2ID["controls"],
        RELATION2ID["prevents"],
        RELATION2ID["treats"],
    ]
    assoc_idx = [RELATION2ID["statistical_association"]]
    no_assoc_idx = [RELATION2ID["no_statistical_association"]]

    n_samples = y_binary.shape[0]
    coarse = np.zeros((n_samples, 4), dtype=int)

    coarse[:, 0] = np.any(y_binary[:, pos_idx] == 1, axis=1).astype(int)
    coarse[:, 1] = np.any(y_binary[:, neg_idx] == 1, axis=1).astype(int)
    coarse[:, 2] = np.any(y_binary[:, assoc_idx] == 1, axis=1).astype(int)
    coarse[:, 3] = np.any(y_binary[:, no_assoc_idx] == 1, axis=1).astype(int)

    return coarse


def compute_coarse_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes evaluation metrics at the coarse level (Positive / Negative / Association / No Assoc).
    """
    coarse_names = ["positive_group", "negative_group", "statistical_association", "no_statistical_association"]
    c_true = map_to_coarse_grained(y_true)
    c_pred = map_to_coarse_grained(y_pred)
    return compute_multilabel_metrics(c_true, c_pred, class_names=coarse_names)


def format_metrics_table(metrics_dict: Dict[str, Any]) -> str:
    """
    Returns a formatted Markdown table of evaluation metrics.
    """
    lines = [
        "| Relation Type | Precision (%) | Recall (%) | F1-Score (%) | Support | Predicted |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for cname, vals in metrics_dict["per_class"].items():
        p = f"{vals['precision'] * 100:.1f}"
        r = f"{vals['recall'] * 100:.1f}"
        f1 = f"{vals['f1'] * 100:.1f}"
        lines.append(f"| **{cname}** | {p}% | {r}% | {f1}% | {vals['support']} | {vals['predicted_count']} |")

    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    m_p = f"{metrics_dict['micro']['precision'] * 100:.1f}"
    m_r = f"{metrics_dict['micro']['recall'] * 100:.1f}"
    m_f1 = f"{metrics_dict['micro']['f1'] * 100:.1f}"
    lines.append(f"| **Micro Average** | **{m_p}%** | **{m_r}%** | **{m_f1}%** | - | - |")

    mac_p = f"{metrics_dict['macro']['precision'] * 100:.1f}"
    mac_r = f"{metrics_dict['macro']['recall'] * 100:.1f}"
    mac_f1 = f"{metrics_dict['macro']['f1'] * 100:.1f}"
    lines.append(f"| **Macro Average** | **{mac_p}%** | **{mac_r}%** | **{mac_f1}%** | - | - |")

    return "\n".join(lines)
