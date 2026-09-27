from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, hamming_loss, precision_recall_fscore_support


@dataclass(frozen=True)
class ThresholdTuningResult:
    thresholds: np.ndarray
    per_class_f1: np.ndarray


def apply_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array [samples, classes]")
    if thresholds.shape != (scores.shape[1],):
        raise ValueError("thresholds must have one value per class")
    return (scores >= thresholds[None, :]).astype(np.int64)


def tune_per_class_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    grid: Iterable[float],
) -> ThresholdTuningResult:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_scores = np.asarray(y_scores, dtype=np.float32)
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have identical shape")
    grid = np.asarray(list(grid), dtype=np.float32)
    if grid.size == 0:
        raise ValueError("threshold grid must not be empty")

    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    best_f1 = np.zeros(y_true.shape[1], dtype=np.float32)
    for class_index in range(y_true.shape[1]):
        class_true = y_true[:, class_index]
        class_scores = y_scores[:, class_index]
        class_best = -1.0
        class_threshold = 0.5
        for threshold in grid:
            pred = (class_scores >= threshold).astype(np.int64)
            _, _, f1, _ = precision_recall_fscore_support(
                class_true,
                pred,
                average="binary",
                zero_division=0,
            )
            # Deterministic tie-break toward 0.5 to avoid unnecessarily extreme thresholds.
            if f1 > class_best + 1e-12 or (
                abs(f1 - class_best) <= 1e-12
                and abs(float(threshold) - 0.5) < abs(float(class_threshold) - 0.5)
            ):
                class_best = float(f1)
                class_threshold = float(threshold)
        thresholds[class_index] = class_threshold
        best_f1[class_index] = max(0.0, class_best)
    return ThresholdTuningResult(thresholds=thresholds, per_class_f1=best_f1)


def multilabel_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_scores = np.asarray(y_scores, dtype=np.float32)
    y_pred = apply_thresholds(y_scores, thresholds)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and predictions must have identical shape")

    metrics: dict[str, object] = {}
    for average in ("micro", "macro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )
        metrics[f"precision_{average}"] = float(precision)
        metrics[f"recall_{average}"] = float(recall)
        metrics[f"f1_{average}"] = float(f1)

    metrics["hamming_loss"] = float(hamming_loss(y_true, y_pred))

    ap_per_class: list[float] = []
    for class_index in range(y_true.shape[1]):
        labels = y_true[:, class_index]
        if np.unique(labels).size < 2:
            ap_per_class.append(float("nan"))
        else:
            ap_per_class.append(float(average_precision_score(labels, y_scores[:, class_index])))
    finite_ap = [value for value in ap_per_class if np.isfinite(value)]
    metrics["mAP"] = float(np.mean(finite_ap)) if finite_ap else float("nan")

    per_class: dict[str, dict[str, float]] = {}
    names = class_names or [str(i) for i in range(y_true.shape[1])]
    for class_index, class_name in enumerate(names):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, class_index],
            y_pred[:, class_index],
            average="binary",
            zero_division=0,
        )
        per_class[str(class_name)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "average_precision": float(ap_per_class[class_index]),
            "threshold": float(thresholds[class_index]),
        }
    metrics["per_class"] = per_class
    return metrics


def rejection_metrics(
    known_scores: np.ndarray,
    ood_scores: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float]:
    known_pred = apply_thresholds(known_scores, thresholds)
    ood_pred = apply_thresholds(ood_scores, thresholds)
    known_rejected = np.sum(known_pred, axis=1) == 0
    ood_rejected = np.sum(ood_pred, axis=1) == 0
    return {
        "known_false_rejection_rate": float(np.mean(known_rejected)) if len(known_rejected) else float("nan"),
        "ood_recall_rejected": float(np.mean(ood_rejected)) if len(ood_rejected) else float("nan"),
        "ood_false_acceptance_rate": float(np.mean(~ood_rejected)) if len(ood_rejected) else float("nan"),
    }


def group_metrics(
    metadata: list[dict],
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: np.ndarray,
    class_names: list[str],
) -> dict[str, dict]:
    """Compute metrics for single/mix and controlled mixture conditions."""
    if len(metadata) != len(y_true):
        raise ValueError("metadata length must match number of samples")
    groups: dict[str, list[int]] = {}
    for idx, item in enumerate(metadata):
        sample_type = str(item.get("sample_type", "unknown"))
        groups.setdefault(f"sample_type={sample_type}", []).append(idx)
        if sample_type == "mix":
            relative_db = item.get("relative_db")
            overlap_ratio = item.get("overlap_ratio")
            groups.setdefault(f"relative_db={float(relative_db):g}", []).append(idx)
            groups.setdefault(f"overlap_ratio={float(overlap_ratio):g}", []).append(idx)

    result: dict[str, dict] = {}
    for name, indices in groups.items():
        index_array = np.asarray(indices, dtype=np.int64)
        result[name] = multilabel_metrics(
            y_true[index_array],
            y_scores[index_array],
            thresholds,
            class_names,
        )
        result[name]["n_samples"] = int(len(indices))
    return result
