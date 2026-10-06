from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .experiments import (
    EXPERIMENT_FREEZE_VERSION,
    OFFICIAL_SEEDS,
    file_sha256,
)


@dataclass(frozen=True)
class DeploymentSelection:
    model_name: str
    seed: int
    best_validation_mAP: float
    run_dir: str
    checkpoint: str
    thresholds: str
    checkpoint_sha256: str
    threshold_artifact_sha256: str
    selection_rule: str

    def as_dict(self) -> dict:
        return asdict(self)


def _load_json(path: Path) -> dict | list:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def select_frozen_deployment(
    experiment_root: str | Path,
    *,
    model_name: str = "crnn",
) -> DeploymentSelection:
    """Choose one frozen run using validation performance only.

    The seed is selected by highest ``best_validation_mAP`` from the already
    recorded training index. Held-out evaluation artifacts are intentionally
    not read here. Ties are resolved by the lower seed.
    """

    root = Path(experiment_root)
    training_index_path = root / "training_index.json"
    freeze_path = root / "experiment_freeze.json"

    training_records = _load_json(training_index_path)
    freeze = _load_json(freeze_path)
    if not isinstance(training_records, list):
        raise ValueError("training_index.json must contain a list")
    if not isinstance(freeze, dict):
        raise ValueError("experiment_freeze.json must contain an object")
    if freeze.get("freeze_version") != EXPERIMENT_FREEZE_VERSION:
        raise ValueError(
            "Unexpected experiment freeze version: "
            f"{freeze.get('freeze_version')!r}"
        )

    candidates: list[tuple[float, int, dict]] = []
    for record in training_records:
        if str(record.get("model")) != model_name:
            continue
        seed = int(record["seed"])
        summary = record.get("training", {})
        score = float(summary.get("best_validation_mAP", float("nan")))
        if not math.isfinite(score):
            raise ValueError(
                f"{model_name} seed {seed}: best_validation_mAP is not finite"
            )
        candidates.append((score, seed, record))

    if not candidates:
        raise ValueError(f"No training records found for model {model_name!r}")

    seeds = tuple(sorted(seed for _, seed, _ in candidates))
    if seeds != OFFICIAL_SEEDS:
        raise ValueError(
            f"{model_name}: expected frozen seeds {list(OFFICIAL_SEEDS)}, "
            f"got {list(seeds)}"
        )

    # Highest validation mAP; lower seed is deterministic tie-break.
    score, seed, record = sorted(
        candidates,
        key=lambda item: (-item[0], item[1]),
    )[0]

    frozen_by_key = {
        (str(item["model"]), int(item["seed"])): item
        for item in freeze.get("runs", [])
    }
    key = (model_name, seed)
    if key not in frozen_by_key:
        raise ValueError(f"Selected deployment run is missing from freeze: {key}")

    frozen = frozen_by_key[key]
    run_dir = Path(record.get("run_dir", root / f"{model_name}_seed{seed}"))
    if not run_dir.is_absolute():
        # training_index normally stores a path relative to repository root.
        # If it no longer resolves, fall back to the canonical experiment path.
        if not run_dir.exists():
            run_dir = root / f"{model_name}_seed{seed}"

    checkpoint = run_dir / "best_model.pt"
    thresholds = run_dir / "thresholds.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Deployment checkpoint not found: {checkpoint}")
    if not thresholds.exists():
        raise FileNotFoundError(f"Deployment thresholds not found: {thresholds}")

    checkpoint_sha = file_sha256(checkpoint)
    threshold_sha = file_sha256(thresholds)
    if checkpoint_sha != frozen.get("checkpoint_sha256"):
        raise ValueError("Selected checkpoint no longer matches experiment freeze")
    if threshold_sha != frozen.get("threshold_artifact_sha256"):
        raise ValueError("Selected threshold artifact no longer matches experiment freeze")

    return DeploymentSelection(
        model_name=model_name,
        seed=seed,
        best_validation_mAP=score,
        run_dir=str(run_dir),
        checkpoint=str(checkpoint),
        thresholds=str(thresholds),
        checkpoint_sha256=checkpoint_sha,
        threshold_artifact_sha256=threshold_sha,
        selection_rule=(
            "highest best_validation_mAP among frozen model-family seeds; "
            "tie -> lower seed"
        ),
    )
