from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .models import model_from_spec


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_spec: dict,
    class_names: list[str],
    feature_config: dict,
    project_config: dict,
    training_metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_spec": model_spec,
        "class_names": list(class_names),
        "feature_config": dict(feature_config),
        "project_config": dict(project_config),
        "training_metadata": training_metadata or {},
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    required = {
        "model_spec",
        "class_names",
        "feature_config",
        "project_config",
        "state_dict",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint missing fields: {sorted(missing)}")
    model = model_from_spec(payload["model_spec"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def _validate_threshold_values(values) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Thresholds must be a non-empty 1D sequence")
    if not np.isfinite(array).all():
        raise ValueError("Thresholds contain non-finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("Thresholds must be within [0, 1]")
    return [float(value) for value in array]


def save_thresholds(
    path: str | Path,
    class_names: list[str],
    thresholds,
    metadata: dict | None = None,
    *,
    overwrite: bool = True,
) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Threshold artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    names = [str(name) for name in class_names]
    values = _validate_threshold_values(thresholds)
    if len(names) != len(values):
        raise ValueError("Threshold count must match class count")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")

    payload = {
        "format_version": 2,
        "class_names": names,
        "thresholds": values,
        "metadata": metadata or {},
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_threshold_artifact(
    path: str | Path,
    expected_classes: list[str] | None = None,
) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Threshold artifact not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Threshold artifact must contain a JSON object")
    required = {"class_names", "thresholds"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Threshold artifact missing fields: {sorted(missing)}")

    class_names = [str(name) for name in payload["class_names"]]
    thresholds = _validate_threshold_values(payload["thresholds"])
    if len(class_names) != len(thresholds):
        raise ValueError("Threshold file has inconsistent lengths")
    if len(set(class_names)) != len(class_names):
        raise ValueError("Threshold artifact class_names must be unique")
    if expected_classes is not None and class_names != list(expected_classes):
        raise ValueError("Threshold class order does not match checkpoint class order")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Threshold artifact metadata must be a JSON object")

    return {
        **payload,
        "format_version": int(payload.get("format_version", 1)),
        "class_names": class_names,
        "thresholds": thresholds,
        "metadata": metadata,
    }


def load_thresholds(
    path: str | Path,
    expected_classes: list[str] | None = None,
) -> list[float]:
    return list(
        load_threshold_artifact(
            path,
            expected_classes=expected_classes,
        )["thresholds"]
    )
