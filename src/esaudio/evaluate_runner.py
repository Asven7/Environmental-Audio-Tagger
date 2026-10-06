from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from .checkpoints import (
    load_checkpoint,
    load_threshold_artifact,
    save_thresholds,
)
from .dataset import AudioManifestDataset
from .evaluation import (
    group_metrics,
    multilabel_metrics,
    rejection_metrics,
    tune_per_class_thresholds,
)
from .features import LogMelExtractor
from .training import predict_dataset, resolve_device


THRESHOLD_PROTOCOL_VERSION = "per_class_validation_f1_v1"
FROZEN_EVALUATION_PROTOCOL_VERSION = "window_multilabel_frozen_eval_v1"


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_threshold_provenance(
    threshold_payload: dict,
    checkpoint_path: str | Path,
    validation_manifest: str | Path,
) -> None:
    metadata = threshold_payload.get("metadata", {})
    required = {
        "protocol_version",
        "selection_split",
        "selection_metric",
        "checkpoint_sha256",
        "validation_manifest_sha256",
        "threshold_grid",
        "validation_samples",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(
            "Threshold artifact lacks frozen-evaluation provenance fields: "
            f"{sorted(missing)}"
        )

    if metadata["protocol_version"] != THRESHOLD_PROTOCOL_VERSION:
        raise ValueError(
            "Unsupported threshold protocol version: "
            f"{metadata['protocol_version']!r}"
        )
    if metadata["selection_split"] != "val":
        raise ValueError("Thresholds must be selected on the validation split")
    if metadata["selection_metric"] != "per_class_f1":
        raise ValueError("Threshold selection metric must be per_class_f1")

    actual_checkpoint_hash = file_sha256(checkpoint_path)
    if metadata["checkpoint_sha256"] != actual_checkpoint_hash:
        raise ValueError(
            "Threshold artifact does not belong to this checkpoint "
            "(checkpoint SHA-256 mismatch)"
        )

    actual_val_hash = file_sha256(validation_manifest)
    if metadata["validation_manifest_sha256"] != actual_val_hash:
        raise ValueError(
            "Threshold artifact was not selected from this validation manifest "
            "(validation manifest SHA-256 mismatch)"
        )


def select_thresholds_from_validation(
    config: dict,
    checkpoint_path: str | Path,
    validation_manifest: str | Path,
    audio_root: str | Path,
    output_path: str | Path,
    *,
    device_name: str = "cpu",
    overwrite: bool = False,
) -> dict:
    """Select and freeze per-class thresholds using known validation data only."""

    device = resolve_device(device_name)
    checkpoint_path = Path(checkpoint_path)
    validation_manifest = Path(validation_manifest)
    output_path = Path(output_path)

    model, checkpoint_payload = load_checkpoint(checkpoint_path, device=device)
    class_names = list(checkpoint_payload["class_names"])
    extractor = LogMelExtractor(**checkpoint_payload["feature_config"]).to(device)

    val_dataset = AudioManifestDataset(
        validation_manifest,
        audio_root,
        config,
        split="val",
    )
    if len(val_dataset) == 0:
        raise ValueError("Validation manifest produced an empty known validation dataset")

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    criterion = nn.BCEWithLogitsLoss()
    val_bundle = predict_dataset(
        model,
        extractor,
        val_loader,
        criterion,
        device,
    )

    threshold_grid = [float(x) for x in config["evaluation"]["threshold_grid"]]
    tuning = tune_per_class_thresholds(
        val_bundle.targets,
        val_bundle.scores,
        threshold_grid,
    )
    tuned_metrics = multilabel_metrics(
        val_bundle.targets,
        val_bundle.scores,
        tuning.thresholds,
        class_names,
    )

    training_metadata = checkpoint_payload.get("training_metadata", {})
    metadata = {
        "protocol_version": THRESHOLD_PROTOCOL_VERSION,
        "selection_split": "val",
        "selection_metric": "per_class_f1",
        "tie_break": "closest_to_0.5_then_lower_threshold",
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "validation_manifest_sha256": file_sha256(validation_manifest),
        "validation_samples": int(len(val_dataset)),
        "threshold_grid": threshold_grid,
        "per_class_validation_f1": [float(x) for x in tuning.per_class_f1],
        "model_name": str(checkpoint_payload["model_spec"]["name"]),
        "experiment_seed": training_metadata.get(
            "experiment_seed",
            training_metadata.get("seed"),
        ),
        "best_epoch": training_metadata.get("best_epoch"),
    }

    save_thresholds(
        output_path,
        class_names,
        tuning.thresholds,
        metadata,
        overwrite=overwrite,
    )

    return {
        "protocol_version": THRESHOLD_PROTOCOL_VERSION,
        "class_names": class_names,
        "thresholds": [float(x) for x in tuning.thresholds],
        "per_class_validation_f1": [float(x) for x in tuning.per_class_f1],
        "validation_metrics_tuned_thresholds": tuned_metrics,
        "validation_loss_unweighted_bce": float(val_bundle.mean_loss),
        "validation_samples": int(len(val_dataset)),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "validation_manifest_sha256": metadata["validation_manifest_sha256"],
        "threshold_artifact_sha256": file_sha256(output_path),
    }


def evaluate_checkpoint(
    config: dict,
    checkpoint_path: str | Path,
    thresholds_path: str | Path,
    known_manifest: str | Path,
    audio_root: str | Path,
    split: str = "test",
    ood_manifest: str | Path | None = None,
    device_name: str = "cpu",
    *,
    require_threshold_provenance: bool = False,
    validation_manifest_for_thresholds: str | Path | None = None,
) -> dict:
    device = resolve_device(device_name)
    checkpoint_path = Path(checkpoint_path)
    thresholds_path = Path(thresholds_path)
    known_manifest = Path(known_manifest)

    model, payload = load_checkpoint(checkpoint_path, device=device)
    class_names = list(payload["class_names"])
    threshold_payload = load_threshold_artifact(
        thresholds_path,
        expected_classes=class_names,
    )
    thresholds = np.asarray(threshold_payload["thresholds"], dtype=np.float32)

    if require_threshold_provenance:
        if validation_manifest_for_thresholds is None:
            raise ValueError(
                "validation_manifest_for_thresholds is required for strict frozen evaluation"
            )
        validate_threshold_provenance(
            threshold_payload,
            checkpoint_path,
            validation_manifest_for_thresholds,
        )

    extractor = LogMelExtractor(**payload["feature_config"]).to(device)
    criterion = nn.BCEWithLogitsLoss()

    known_dataset = AudioManifestDataset(
        known_manifest,
        audio_root,
        config,
        split=split,
    )
    if len(known_dataset) == 0:
        raise ValueError(
            f"Known manifest produced an empty dataset for split={split!r}: {known_manifest}"
        )

    known_loader = DataLoader(
        known_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    known_bundle = predict_dataset(
        model,
        extractor,
        known_loader,
        criterion,
        device,
    )

    provenance = {
        "protocol_version": (
            FROZEN_EVALUATION_PROTOCOL_VERSION
            if require_threshold_provenance
            else "generic_evaluation"
        ),
        "split": split,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "threshold_artifact_sha256": file_sha256(thresholds_path),
        "known_manifest_sha256": file_sha256(known_manifest),
        "validation_manifest_sha256": (
            file_sha256(validation_manifest_for_thresholds)
            if validation_manifest_for_thresholds is not None
            else None
        ),
        "class_names": class_names,
        "thresholds": [float(x) for x in thresholds],
        "threshold_metadata": threshold_payload.get("metadata", {}),
        "model_name": str(payload["model_spec"]["name"]),
        "experiment_seed": payload.get("training_metadata", {}).get(
            "experiment_seed",
            payload.get("training_metadata", {}).get("seed"),
        ),
    }

    result = {
        "known": multilabel_metrics(
            known_bundle.targets,
            known_bundle.scores,
            thresholds,
            class_names,
        ),
        "groups": group_metrics(
            known_bundle.metadata,
            known_bundle.targets,
            known_bundle.scores,
            thresholds,
            class_names,
        ),
        "known_loss_unweighted_bce": float(known_bundle.mean_loss),
        "known_loss": float(known_bundle.mean_loss),
        "n_known": int(len(known_dataset)),
        "provenance": provenance,
    }

    if ood_manifest is not None:
        ood_manifest = Path(ood_manifest)
        if not ood_manifest.exists():
            raise FileNotFoundError(f"OOD manifest not found: {ood_manifest}")

        ood_dataset = AudioManifestDataset(
            ood_manifest,
            audio_root,
            config,
            split=split,
            include_ood=True,
        )
        if len(ood_dataset) == 0:
            raise ValueError(
                f"OOD manifest produced an empty dataset for split={split!r}: {ood_manifest}"
            )

        ood_loader = DataLoader(
            ood_dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=0,
        )
        ood_bundle = predict_dataset(
            model,
            extractor,
            ood_loader,
            criterion,
            device,
        )
        result["rejection"] = rejection_metrics(
            known_bundle.scores,
            ood_bundle.scores,
            thresholds,
        )
        result["n_ood"] = int(len(ood_dataset))
        result["provenance"]["ood_manifest_sha256"] = file_sha256(ood_manifest)
    else:
        result["provenance"]["ood_manifest_sha256"] = None

    return result


def save_evaluation(
    result: dict,
    path: str | Path,
    *,
    overwrite: bool = True,
) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Evaluation artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
