from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


_REQUIRED_PATHS = [
    ("project", "sample_rate"),
    ("project", "window_seconds"),
    ("project", "hop_seconds"),
    ("project", "target_classes"),
    ("project", "heldout_classes"),
    ("features", "n_fft"),
    ("features", "hop_length"),
    ("features", "n_mels"),
    ("model", "cnn_channels"),
    ("training", "batch_size"),
    ("training", "epochs"),
    ("training", "learning_rate"),
    ("evaluation", "threshold_grid"),
]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, base_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration, optionally merging it over a base YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if base_path is not None:
        base_path = Path(base_path)
        with base_path.open("r", encoding="utf-8") as handle:
            base = yaml.safe_load(handle) or {}
        data = _deep_merge(base, data)

    validate_config(data)
    return data


def validate_config(config: dict[str, Any]) -> None:
    for section, key in _REQUIRED_PATHS:
        if section not in config or key not in config[section]:
            raise ValueError(f"Missing required configuration key: {section}.{key}")

    project = config["project"]
    sr = int(project["sample_rate"])
    window_seconds = float(project["window_seconds"])
    hop_seconds = float(project["hop_seconds"])
    target_classes = list(project["target_classes"])
    heldout_classes = list(project["heldout_classes"])

    if sr <= 0:
        raise ValueError("project.sample_rate must be positive")
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")
    if hop_seconds > window_seconds:
        raise ValueError("hop_seconds must not exceed window_seconds")
    if len(target_classes) < 2:
        raise ValueError("At least two target classes are required")
    if len(set(target_classes)) != len(target_classes):
        raise ValueError("target_classes must be unique")
    if set(target_classes) & set(heldout_classes):
        raise ValueError("target_classes and heldout_classes must be disjoint")

    features = config["features"]
    n_fft = int(features["n_fft"])
    feature_hop = int(features["hop_length"])
    n_mels = int(features["n_mels"])
    if n_fft <= 0 or feature_hop <= 0 or n_mels <= 0:
        raise ValueError("Feature parameters must be positive")
    if feature_hop > n_fft:
        raise ValueError("features.hop_length should not exceed features.n_fft")

    channels = config["model"]["cnn_channels"]
    if not channels or any(int(c) <= 0 for c in channels):
        raise ValueError("model.cnn_channels must contain positive channel counts")

    training = config["training"]
    if int(training["batch_size"]) <= 0 or int(training["epochs"]) <= 0:
        raise ValueError("training.batch_size and training.epochs must be positive")
    if float(training["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be positive")

    grid = [float(x) for x in config["evaluation"]["threshold_grid"]]
    if not grid or any(x <= 0 or x >= 1 for x in grid):
        raise ValueError("evaluation.threshold_grid must contain values strictly between 0 and 1")


def window_samples(config: dict[str, Any]) -> int:
    return round(float(config["project"]["window_seconds"]) * int(config["project"]["sample_rate"]))


def stream_hop_samples(config: dict[str, Any]) -> int:
    return round(float(config["project"]["hop_seconds"]) * int(config["project"]["sample_rate"]))
