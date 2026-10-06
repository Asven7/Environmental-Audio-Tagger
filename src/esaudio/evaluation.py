from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    hamming_loss,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class ThresholdTuningResult:
    thresholds: np.ndarray
    per_class_f1: np.ndarray


def _validated_scores_and_targets(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=np.float32)

    if y_true.ndim != 2 or y_scores.ndim != 2:
        raise ValueError("y_true and y_scores must be 2D [samples, classes] arrays")
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have identical shape")
    if y_true.shape[0] == 0 or y_true.shape[1] == 0:
        raise ValueError("y_true and y_scores must be non-empty")
    if not np.isfinite(y_scores).all():
        raise ValueError("y_scores contains non-finite values")

    y_true = y_true.astype(np.int64, copy=False)
    if not np.logical_or(y_true == 0, y_true == 1).all():
        raise ValueError("y_true must contain binary 0/1 labels")
    return y_true, y_scores


def _validated_threshold_grid(grid: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(grid), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("threshold grid must be a non-empty 1D sequence")
    if not np.isfinite(values).all():
        raise ValueError("threshold grid contains non-finite values")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("threshold grid values must be strictly between 0 and 1")
    return np.unique(values)


def apply_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array [samples, classes]")
    if not np.isfinite(scores).all():
        raise ValueError("scores contains non-finite values")
    if thresholds.shape != (scores.shape[1],):
        raise ValueError("thresholds must have one value per class")
    if not np.isfinite(thresholds).all():
        raise ValueError("thresholds contains non-finite values")
    if np.any(thresholds < 0.0) or np.any(thresholds > 1.0):
        raise ValueError("thresholds must be within [0, 1]")
    return (scores >= thresholds[None, :]).astype(np.int64)


def tune_per_class_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    grid: Iterable[float],
) -> ThresholdTuningResult:
    """Tune one threshold per class by validation F1 only.

    Each validation class must contain at least one positive and one negative example.
    Ties are resolved deterministically by:
      1) threshold closest to 0.5;
      2) lower threshold if distances to 0.5 are identical.
    """

    y_true, y_scores = _validated_scores_and_targets(y_true, y_scores)
    grid_values = _validated_threshold_grid(grid)

    positives = y_true.sum(axis=0)
    negatives = y_true.shape[0] - positives
    missing_positive = np.flatnonzero(positives <= 0)
    missing_negative = np.flatnonzero(negatives <= 0)
    if missing_positive.size:
        raise ValueError(
            "Threshold tuning requires at least one positive validation example "
            f"for every class; missing positive class indices {missing_positive.tolist()}"
        )
    if missing_negative.size:
        raise ValueError(
            "Threshold tuning requires at least one negative validation example "
            f"for every class; missing negative class indices {missing_negative.tolist()}"
        )

    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    best_f1 = np.zeros(y_true.shape[1], dtype=np.float32)

    for class_index in range(y_true.shape[1]):
        class_true = y_true[:, class_index]
        class_scores = y_scores[:, class_index]
        candidates: list[tuple[float, float]] = []

        for threshold in grid_values:
            pred = (class_scores >= threshold).astype(np.int64)
            _, _, f1, _ = precision_recall_fscore_support(
                class_true,
                pred,
                average="binary",
                zero_division=0,
            )
            candidates.append((float(f1), float(threshold)))

        max_f1 = max(item[0] for item in candidates)
        tied = [
            threshold
            for f1, threshold in candidates
            if abs(f1 - max_f1) <= 1e-12
        ]
        class_threshold = min(
            tied,
            key=lambda threshold: (abs(threshold - 0.5), threshold),
        )

        thresholds[class_index] = float(class_threshold)
        best_f1[class_index] = float(max_f1)

    return ThresholdTuningResult(
        thresholds=thresholds,
        per_class_f1=best_f1,
    )


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
    if class_names is not None and len(class_names) != y_true.shape[1]:
        raise ValueError("class_names length must match the number of classes")

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
            ap_per_class.append(
                float(average_precision_score(labels, y_scores[:, class_index]))
            )
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

    known_false_rejection = (
        float(np.mean(known_rejected)) if len(known_rejected) else float("nan")
    )
    ood_rejection = (
        float(np.mean(ood_rejected)) if len(ood_rejected) else float("nan")
    )
    ood_false_acceptance = (
        float(np.mean(~ood_rejected)) if len(ood_rejected) else float("nan")
    )

    return {
        "known_false_rejection_rate": known_false_rejection,
        "known_acceptance_rate": (
            1.0 - known_false_rejection
            if np.isfinite(known_false_rejection)
            else float("nan")
        ),
        "ood_rejection_rate": ood_rejection,
        "ood_recall_rejected": ood_rejection,
        "ood_false_acceptance_rate": ood_false_acceptance,
    }


def group_metrics(
    metadata: list[dict],
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: np.ndarray,
    class_names: list[str],
) -> dict[str, dict]:
    """Compute metrics for singles, mixtures, and controlled mixture conditions."""

    if len(metadata) != len(y_true):
        raise ValueError("metadata length must match number of samples")
    groups: dict[str, list[int]] = {}

    for idx, item in enumerate(metadata):
        sample_type = str(item.get("sample_type", "unknown"))
        groups.setdefault(f"sample_type={sample_type}", []).append(idx)

        if sample_type == "mix":
            relative_db = float(item.get("relative_db"))
            overlap_ratio = float(item.get("overlap_ratio"))
            groups.setdefault(f"relative_db={relative_db:g}", []).append(idx)
            groups.setdefault(f"overlap_ratio={overlap_ratio:g}", []).append(idx)
            groups.setdefault(
                f"mix_condition=relative_db={relative_db:g}|overlap_ratio={overlap_ratio:g}",
                [],
            ).append(idx)

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
