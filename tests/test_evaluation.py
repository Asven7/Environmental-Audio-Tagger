import numpy as np

from esaudio.evaluation import apply_thresholds, multilabel_metrics, tune_per_class_thresholds


def test_threshold_tuning_and_metrics():
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=int)
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.7], [0.1, 0.2]], dtype=float)
    tuned = tune_per_class_thresholds(y_true, scores, [0.3, 0.5, 0.7])
    pred = apply_thresholds(scores, tuned.thresholds)
    assert np.array_equal(pred, y_true)
    metrics = multilabel_metrics(y_true, scores, tuned.thresholds, ["a", "b"])
    assert metrics["f1_micro"] == 1.0
    assert metrics["f1_macro"] == 1.0
