from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


_REQUIRED_SECTIONS = (
    "project",
    "data",
    "features",
    "model",
    "training",
    "augmentation",
    "mixing",
    "evaluation",
)
_CROP_STRATEGIES = {"start", "center", "energy"}
_RECURRENT_TYPES = {"gru", "lstm"}


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def _require_positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"'{name}' must be a positive number")
    return float(value)


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"'{name}' must be a positive integer")
    return value


def _require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"'{name}' must be a non-negative integer")
    return value


def _validate_class_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{name}' must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"'{name}' must contain non-empty strings")
    normalized = [item.strip() for item in value]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"'{name}' must not contain duplicate class names")
    return normalized


def _validate_fold_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{name}' must be a non-empty list")
    if not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value):
        raise ValueError(f"'{name}' must contain positive integer fold numbers")
    if len(value) != len(set(value)):
        raise ValueError(f"'{name}' must not contain duplicate folds")
    return list(value)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the project configuration and return a normalized deep copy.

    Validation is intentionally strict for values that affect experimental
    reproducibility (splits, windowing, feature dimensions, and thresholds).
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")

    normalized = deepcopy(config)
    for section in _REQUIRED_SECTIONS:
        if section not in normalized:
            raise ValueError(f"Missing required configuration section: '{section}'")
        _require_mapping(normalized[section], section)

    project = normalized["project"]
    data = normalized["data"]
    features = normalized["features"]
    model = normalized["model"]
    training = normalized["training"]
    augmentation = normalized["augmentation"]
    mixing = normalized["mixing"]
    evaluation = normalized["evaluation"]

    project["sample_rate"] = _require_positive_int(project.get("sample_rate"), "project.sample_rate")
    project["window_seconds"] = _require_positive_number(
        project.get("window_seconds"), "project.window_seconds"
    )
    project["hop_seconds"] = _require_positive_number(project.get("hop_seconds"), "project.hop_seconds")
    if project["hop_seconds"] > project["window_seconds"]:
        raise ValueError("'project.hop_seconds' must not exceed 'project.window_seconds'")

    project["target_classes"] = _validate_class_list(project.get("target_classes"), "project.target_classes")
    project["heldout_classes"] = _validate_class_list(
        project.get("heldout_classes"), "project.heldout_classes"
    )
    overlap = set(project["target_classes"]) & set(project["heldout_classes"])
    if overlap:
        raise ValueError(f"Target and held-out classes must be disjoint; overlap: {sorted(overlap)}")

    data["train_folds"] = _validate_fold_list(data.get("train_folds"), "data.train_folds")
    data["val_folds"] = _validate_fold_list(data.get("val_folds"), "data.val_folds")
    data["test_folds"] = _validate_fold_list(data.get("test_folds"), "data.test_folds")
    fold_sets = [set(data[key]) for key in ("train_folds", "val_folds", "test_folds")]
    if fold_sets[0] & fold_sets[1] or fold_sets[0] & fold_sets[2] or fold_sets[1] & fold_sets[2]:
        raise ValueError("Train/validation/test fold assignments must be disjoint")

    for key in ("train_crop_strategy", "eval_crop_strategy"):
        value = data.get(key)
        if value not in _CROP_STRATEGIES:
            raise ValueError(f"'data.{key}' must be one of {sorted(_CROP_STRATEGIES)}")
    if isinstance(data.get("target_rms_dbfs"), bool) or not isinstance(data.get("target_rms_dbfs"), (int, float)):
        raise ValueError("'data.target_rms_dbfs' must be numeric")
    data["target_rms_dbfs"] = float(data["target_rms_dbfs"])

    features["n_fft"] = _require_positive_int(features.get("n_fft"), "features.n_fft")
    features["hop_length"] = _require_positive_int(features.get("hop_length"), "features.hop_length")
    if features["hop_length"] > features["n_fft"]:
        raise ValueError("'features.hop_length' must not exceed 'features.n_fft'")
    features["n_mels"] = _require_positive_int(features.get("n_mels"), "features.n_mels")
    features["f_min"] = float(features.get("f_min", 0.0))
    features["f_max"] = float(features.get("f_max", project["sample_rate"] / 2))
    if features["f_min"] < 0:
        raise ValueError("'features.f_min' must be non-negative")
    nyquist = project["sample_rate"] / 2.0
    if not features["f_min"] < features["f_max"] <= nyquist:
        raise ValueError(
            f"Feature frequency range must satisfy 0 <= f_min < f_max <= Nyquist ({nyquist:g} Hz)"
        )
    if not isinstance(features.get("normalize_features"), bool):
        raise ValueError("'features.normalize_features' must be boolean")

    channels = model.get("cnn_channels")
    if not isinstance(channels, list) or not channels or not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in channels
    ):
        raise ValueError("'model.cnn_channels' must be a non-empty list of positive integers")
    recurrent_type = str(model.get("recurrent_type", "")).lower()
    if recurrent_type not in _RECURRENT_TYPES:
        raise ValueError(f"'model.recurrent_type' must be one of {sorted(_RECURRENT_TYPES)}")
    model["recurrent_type"] = recurrent_type
    model["recurrent_hidden"] = _require_positive_int(model.get("recurrent_hidden"), "model.recurrent_hidden")
    model["recurrent_layers"] = _require_positive_int(model.get("recurrent_layers"), "model.recurrent_layers")
    dropout = model.get("dropout")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= float(dropout) < 1:
        raise ValueError("'model.dropout' must be in [0, 1)")
    model["dropout"] = float(dropout)

    training["batch_size"] = _require_positive_int(training.get("batch_size"), "training.batch_size")
    training["epochs"] = _require_positive_int(training.get("epochs"), "training.epochs")
    training["learning_rate"] = _require_positive_number(
        training.get("learning_rate"), "training.learning_rate"
    )
    if isinstance(training.get("weight_decay"), bool) or not isinstance(
        training.get("weight_decay"), (int, float)
    ) or float(training["weight_decay"]) < 0:
        raise ValueError("'training.weight_decay' must be non-negative")
    training["weight_decay"] = float(training["weight_decay"])
    training["early_stopping_patience"] = _require_positive_int(
        training.get("early_stopping_patience"), "training.early_stopping_patience"
    )
    if not isinstance(training.get("use_pos_weight"), bool):
        raise ValueError("'training.use_pos_weight' must be boolean")
    training["data_seed"] = _require_non_negative_int(training.get("data_seed"), "training.data_seed")
    seeds = training.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in seeds
    ):
        raise ValueError("'training.seeds' must be a non-empty list of non-negative integers")

    if not isinstance(augmentation.get("enabled"), bool):
        raise ValueError("'augmentation.enabled' must be boolean")
    noise_probability = augmentation.get("noise_probability")
    if isinstance(noise_probability, bool) or not isinstance(noise_probability, (int, float)) or not 0 <= float(noise_probability) <= 1:
        raise ValueError("'augmentation.noise_probability' must be in [0, 1]")

    relative_levels = mixing.get("relative_db_levels")
    if not isinstance(relative_levels, list) or not relative_levels or not all(
        isinstance(level, (int, float)) and not isinstance(level, bool) for level in relative_levels
    ):
        raise ValueError("'mixing.relative_db_levels' must be a non-empty numeric list")
    overlap_ratios = mixing.get("overlap_ratios")
    if not isinstance(overlap_ratios, list) or not overlap_ratios or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < float(value) <= 1
        for value in overlap_ratios
    ):
        raise ValueError("'mixing.overlap_ratios' must contain values in (0, 1]")
    for key in ("train_mixtures", "val_mixtures", "test_mixtures"):
        mixing[key] = _require_non_negative_int(mixing.get(key), f"mixing.{key}")

    threshold_grid = evaluation.get("threshold_grid")
    if not isinstance(threshold_grid, list) or not threshold_grid:
        raise ValueError("'evaluation.threshold_grid' must be a non-empty list")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < float(value) < 1
        for value in threshold_grid
    ):
        raise ValueError("'evaluation.threshold_grid' values must be strictly between 0 and 1")
    evaluation["threshold_grid"] = [float(value) for value in threshold_grid]

    return normalized


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in configuration file '{path}': {exc}") from exc
    return validate_config(raw)


def window_samples(config: dict[str, Any]) -> int:
    project = config["project"]
    return round(int(project["sample_rate"]) * float(project["window_seconds"]))


def stream_hop_samples(config: dict[str, Any]) -> int:
    project = config["project"]
    return round(int(project["sample_rate"]) * float(project["hop_seconds"]))
