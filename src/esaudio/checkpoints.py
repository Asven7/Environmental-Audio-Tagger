from __future__ import annotations

import json
from pathlib import Path

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


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    required = {"model_spec", "class_names", "feature_config", "project_config", "state_dict"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint missing fields: {sorted(missing)}")
    model = model_from_spec(payload["model_spec"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def save_thresholds(path: str | Path, class_names: list[str], thresholds, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [float(x) for x in thresholds]
    payload = {
        "class_names": list(class_names),
        "thresholds": values,
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_thresholds(path: str | Path, expected_classes: list[str] | None = None) -> list[float]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    class_names = list(payload["class_names"])
    thresholds = [float(x) for x in payload["thresholds"]]
    if expected_classes is not None and class_names != list(expected_classes):
        raise ValueError("Threshold class order does not match checkpoint class order")
    if len(class_names) != len(thresholds):
        raise ValueError("Threshold file has inconsistent lengths")
    return thresholds
