from typing import Any, Dict, List

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_classification_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    if not y_true:
        return {
            "n_samples": 0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": {},
            "confusion_matrix": [],
            "labels": labels,
        }

    _, _, f1_micro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        labels=labels,
        zero_division=0,
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        labels=labels,
        zero_division=0,
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        labels=labels,
        zero_division=0,
    )

    per_class = {}
    for label in labels:
        y_true_bin = [1 if x == label else 0 for x in y_true]
        y_pred_bin = [1 if x == label else 0 for x in y_pred]
        p, r, f1, s = precision_recall_fscore_support(
            y_true_bin,
            y_pred_bin,
            average="binary",
            zero_division=0,
        )
        per_class[label] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support": int(s),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "n_samples": len(y_true),
        "micro_f1": float(f1_micro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels,
    }
