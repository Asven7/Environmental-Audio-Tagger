from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoints import load_checkpoint, save_checkpoint, save_thresholds
from .dataset import AudioManifestDataset
from .evaluation import multilabel_metrics, tune_per_class_thresholds
from .features import feature_extractor_from_config
from .models import build_model, count_parameters

LOGGER = logging.getLogger(__name__)


@dataclass
class PredictionBundle:
    targets: np.ndarray
    scores: np.ndarray
    metadata: list[dict]
    mean_loss: float


@dataclass
class EarlyStoppingState:
    """Track a validation metric without consulting the test split."""

    patience: int
    min_delta: float = 1e-6
    best_value: float = -math.inf
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    def update(self, value: float, epoch: int) -> bool:
        improved = bool(np.isfinite(value) and value > self.best_value + self.min_delta)
        if improved:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience


def seed_everything(seed: int) -> None:
    """Seed global RNGs and request deterministic kernels where practical.

    A dedicated DataLoader generator is still used separately so that train-sample
    order does not depend on how many random values a model architecture consumes
    during initialization.
    """

    seed = int(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # warn_only keeps the project usable if a platform lacks a deterministic
    # implementation for a particular operation, while still surfacing a warning.
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_dataloader_generator(seed: int) -> torch.Generator:
    """Return an RNG dedicated to DataLoader shuffling."""

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(requested)


def compute_pos_weight(target_matrix: np.ndarray) -> torch.Tensor:
    """Compute BCE positive-class weights from TRAINING labels only."""

    matrix = np.asarray(target_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("target_matrix must be a non-empty 2D [samples, classes] array")
    if not np.isfinite(matrix).all():
        raise ValueError("target_matrix contains non-finite values")
    if not np.logical_or(matrix == 0.0, matrix == 1.0).all():
        raise ValueError("target_matrix must contain binary 0/1 labels")

    positives = matrix.sum(axis=0)
    missing = np.flatnonzero(positives <= 0.0)
    if missing.size:
        raise ValueError(
            "Cannot compute pos_weight: training data has no positive example for "
            f"class indices {missing.tolist()}"
        )

    negatives = matrix.shape[0] - positives
    weight = negatives / positives
    if not np.isfinite(weight).all():
        raise ValueError("Computed pos_weight contains non-finite values")
    return torch.tensor(weight, dtype=torch.float32)


def _sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment_metadata(device: torch.device) -> dict:
    cuda_device_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name(device)
    cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return {
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cudnn_version": int(cudnn_version) if cudnn_version is not None else None,
        "device": str(device),
        "cuda_device_name": cuda_device_name,
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic)
        if torch.backends.cudnn.is_available()
        else None,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)
        if torch.backends.cudnn.is_available()
        else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _batch_metadata(batch: dict, index: int) -> dict:
    relative = batch["relative_db"]
    overlap = batch["overlap_ratio"]
    rel_value = float(relative[index]) if torch.is_tensor(relative) else float(relative[index])
    overlap_value = float(overlap[index]) if torch.is_tensor(overlap) else float(overlap[index])
    return {
        "sample_id": batch["sample_id"][index],
        "sample_type": batch["sample_type"][index],
        "relative_db": rel_value,
        "overlap_ratio": overlap_value,
    }


def train_one_epoch(
    model: nn.Module,
    extractor: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    extractor.eval()  # deterministic transform; no trainable state
    running_loss = 0.0
    samples = 0
    for batch in loader:
        waveform = batch["waveform"].to(device)
        target = batch["target"].to(device)
        with torch.no_grad():
            features = extractor(waveform)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        batch_size = waveform.shape[0]
        running_loss += float(loss.detach().cpu()) * batch_size
        samples += batch_size
    return running_loss / max(samples, 1)


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    extractor: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> PredictionBundle:
    model.eval()
    extractor.eval()
    all_targets: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    metadata: list[dict] = []
    running_loss = 0.0
    samples = 0
    for batch in loader:
        waveform = batch["waveform"].to(device)
        target = batch["target"].to(device)
        features = extractor(waveform)
        logits = model(features)
        loss = criterion(logits, target)
        scores = torch.sigmoid(logits)
        batch_size = waveform.shape[0]
        all_targets.append(target.cpu().numpy())
        all_scores.append(scores.cpu().numpy())
        running_loss += float(loss.cpu()) * batch_size
        samples += batch_size
        for index in range(batch_size):
            metadata.append(_batch_metadata(batch, index))
    return PredictionBundle(
        targets=np.concatenate(all_targets, axis=0) if all_targets else np.empty((0, 0)),
        scores=np.concatenate(all_scores, axis=0) if all_scores else np.empty((0, 0)),
        metadata=metadata,
        mean_loss=running_loss / max(samples, 1),
    )


def train_model(
    config: dict,
    model_name: str,
    train_manifest: str | Path,
    val_manifest: str | Path,
    audio_root: str | Path,
    output_dir: str | Path,
    seed: int,
    *,
    tune_thresholds: bool = True,
) -> dict:
    """Train using TRAIN + VALIDATION only.

    ``tune_thresholds`` remains True by default for backward compatibility with
    the existing engineering demo pipeline. Research experiment runners must pass
    ``False`` so threshold selection stays in the later frozen-evaluation phase.
    This function never reads a test or OOD manifest.
    """

    seed = int(seed)
    seed_everything(seed)
    device = resolve_device(str(config["training"].get("device", "auto")))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(config["project"]["target_classes"])
    train_manifest = Path(train_manifest)
    val_manifest = Path(val_manifest)

    train_dataset = AudioManifestDataset(train_manifest, audio_root, config, split="train")
    val_dataset = AudioManifestDataset(val_manifest, audio_root, config, split="val")
    if len(train_dataset) == 0:
        raise ValueError("Training manifest produced an empty training dataset")
    if len(val_dataset) == 0:
        raise ValueError("Validation manifest produced an empty validation dataset")

    batch_size = int(config["training"]["batch_size"])
    # A dedicated shuffle RNG keeps data order reproducible and independent of
    # architecture-specific model initialization RNG consumption.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=make_dataloader_generator(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model, model_spec = build_model(config, model_name, len(class_names))
    model.to(device)
    extractor = feature_extractor_from_config(config).to(device)

    use_pos_weight = bool(config["training"].get("use_pos_weight", True))
    pos_weight_cpu = compute_pos_weight(train_dataset.target_matrix()) if use_pos_weight else None
    pos_weight = pos_weight_cpu.to(device) if pos_weight_cpu is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    epochs = int(config["training"]["epochs"])
    patience = int(config["training"].get("early_stopping_patience", 5))
    early_stopping = EarlyStoppingState(patience=patience)
    history: list[dict] = []
    checkpoint_path = output_dir / "best_model.pt"
    stop_reason = "max_epochs"

    reproducibility_contract = {
        "experiment_seed": seed,
        "data_seed": int(config["training"].get("data_seed", 1234)),
        "train_shuffle_seed": seed,
        "num_workers": 0,
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": _sha256_file(train_manifest),
        "val_manifest": str(val_manifest),
        "val_manifest_sha256": _sha256_file(val_manifest),
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
        "augmentation_enabled_for_train": bool(config.get("augmentation", {}).get("enabled", False)),
        "validation_augmentation": False,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "early_stopping_patience": patience,
        "early_stopping_metric": "validation_mAP",
        "use_pos_weight": use_pos_weight,
        "pos_weight": pos_weight_cpu.tolist() if pos_weight_cpu is not None else None,
        "threshold_tuning_requested": bool(tune_thresholds),
        "environment": _environment_metadata(device),
    }

    LOGGER.info(
        "Training %s on %s: %d train samples, %d val samples, %d parameters",
        model_name,
        device,
        len(train_dataset),
        len(val_dataset),
        count_parameters(model),
    )

    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        train_loss = train_one_epoch(
            model, extractor, train_loader, optimizer, criterion, device
        )
        val_bundle = predict_dataset(model, extractor, val_loader, criterion, device)
        default_thresholds = np.full(len(class_names), 0.5, dtype=np.float32)
        val_metrics = multilabel_metrics(
            val_bundle.targets,
            val_bundle.scores,
            default_thresholds,
            class_names,
        )

        # mAP is score/ranking based; the 0.5 thresholds above are only needed
        # for the threshold-dependent reporting metrics in multilabel_metrics().
        val_map = float(val_metrics["mAP"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_bundle.mean_loss,
                "val_mAP": val_map,
                # Keep the old key for compatibility with any existing analysis.
                "val_mAP_at_0_5_reporting": val_map,
                "val_f1_macro_at_0_5": float(val_metrics["f1_macro"]),
            }
        )

        LOGGER.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_mAP=%.4f val_f1_macro=%.4f",
            epoch,
            train_loss,
            val_bundle.mean_loss,
            val_map,
            float(val_metrics["f1_macro"]),
        )

        improved = early_stopping.update(val_map, epoch)
        if improved:
            save_checkpoint(
                checkpoint_path,
                model,
                model_spec.as_dict(),
                class_names,
                extractor.export_config(),
                {
                    "sample_rate": int(config["project"]["sample_rate"]),
                    "window_seconds": float(config["project"]["window_seconds"]),
                    "hop_seconds": float(config["project"]["hop_seconds"]),
                    "target_rms_dbfs": float(
                        config.get("data", {}).get("target_rms_dbfs", -20.0)
                    ),
                },
                {
                    **reproducibility_contract,
                    "best_epoch": epoch,
                    "validation_mAP": val_map,
                    "parameter_count": count_parameters(model),
                },
            )
        elif early_stopping.should_stop:
            stop_reason = "early_stopping"
            LOGGER.info("Early stopping at epoch %d", epoch)
            break

    history_path = output_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    if early_stopping.best_epoch <= 0 or not checkpoint_path.exists():
        raise RuntimeError(
            "Training finished without a finite validation mAP; no best checkpoint "
            "could be selected."
        )

    best_model, _ = load_checkpoint(checkpoint_path, device=device)

    thresholds_path: Path | None = None
    tuned_metrics: dict | None = None
    if tune_thresholds:
        # Backward-compatible engineering/demo behavior. Research runs disable
        # this and defer threshold selection to Phase 10.
        val_bundle = predict_dataset(best_model, extractor, val_loader, criterion, device)
        tuning = tune_per_class_thresholds(
            val_bundle.targets,
            val_bundle.scores,
            config["evaluation"]["threshold_grid"],
        )
        thresholds_path = output_dir / "thresholds.json"
        save_thresholds(
            thresholds_path,
            class_names,
            tuning.thresholds,
            {
                "selection": "per-class F1 maximization on validation known samples",
                "seed": seed,
                "per_class_validation_f1": [float(x) for x in tuning.per_class_f1],
            },
        )
        tuned_metrics = multilabel_metrics(
            val_bundle.targets,
            val_bundle.scores,
            tuning.thresholds,
            class_names,
        )

    summary = {
        "model_name": model_name,
        "seed": seed,
        "data_seed": int(config["training"].get("data_seed", 1234)),
        "device": str(device),
        "parameter_count": count_parameters(best_model),
        "best_epoch": int(early_stopping.best_epoch),
        "epochs_ran": int(len(history)),
        "stop_reason": stop_reason,
        "best_validation_mAP": float(early_stopping.best_value),
        "thresholds_tuned": bool(tune_thresholds),
        "validation_metrics_tuned_thresholds": tuned_metrics,
        "checkpoint": str(checkpoint_path),
        "thresholds": str(thresholds_path) if thresholds_path is not None else None,
        "history": str(history_path),
        "train_manifest_sha256": reproducibility_contract["train_manifest_sha256"],
        "val_manifest_sha256": reproducibility_contract["val_manifest_sha256"],
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
        "pos_weight": reproducibility_contract["pos_weight"],
        "reproducibility": reproducibility_contract,
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
