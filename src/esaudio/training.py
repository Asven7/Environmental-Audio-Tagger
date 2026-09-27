from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoints import save_checkpoint, save_thresholds
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(requested)


def compute_pos_weight(target_matrix: np.ndarray) -> torch.Tensor:
    positives = target_matrix.sum(axis=0)
    negatives = target_matrix.shape[0] - positives
    weight = negatives / np.maximum(positives, 1.0)
    return torch.tensor(weight, dtype=torch.float32)


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
) -> dict:
    seed_everything(seed)
    device = resolve_device(str(config["training"].get("device", "auto")))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(config["project"]["target_classes"])

    train_dataset = AudioManifestDataset(train_manifest, audio_root, config, split="train")
    val_dataset = AudioManifestDataset(val_manifest, audio_root, config, split="val")
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model, model_spec = build_model(config, model_name, len(class_names))
    model.to(device)
    extractor = feature_extractor_from_config(config).to(device)
    use_pos_weight = bool(config["training"].get("use_pos_weight", True))
    pos_weight = compute_pos_weight(train_dataset.target_matrix()).to(device) if use_pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    epochs = int(config["training"]["epochs"])
    patience = int(config["training"].get("early_stopping_patience", 5))
    best_map = -math.inf
    epochs_without_improvement = 0
    history: list[dict] = []
    checkpoint_path = output_dir / "best_model.pt"

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
        train_loss = train_one_epoch(model, extractor, train_loader, optimizer, criterion, device)
        val_bundle = predict_dataset(model, extractor, val_loader, criterion, device)
        default_thresholds = np.full(len(class_names), 0.5, dtype=np.float32)
        val_metrics = multilabel_metrics(
            val_bundle.targets,
            val_bundle.scores,
            default_thresholds,
            class_names,
        )
        val_map = float(val_metrics["mAP"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_bundle.mean_loss,
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

        if np.isfinite(val_map) and val_map > best_map + 1e-6:
            best_map = val_map
            epochs_without_improvement = 0
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
                    "target_rms_dbfs": float(config.get("data", {}).get("target_rms_dbfs", -20.0)),
                },
                {
                    "seed": seed,
                    "best_epoch": epoch,
                    "validation_mAP": val_map,
                    "parameter_count": count_parameters(model),
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info("Early stopping at epoch %d", epoch)
                break

    history_path = output_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    # Reload best model for threshold tuning to avoid accidentally using the final epoch.
    from .checkpoints import load_checkpoint

    best_model, _ = load_checkpoint(checkpoint_path, device=device)
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
        "seed": int(seed),
        "device": str(device),
        "parameter_count": count_parameters(best_model),
        "best_validation_mAP": float(best_map),
        "validation_metrics_tuned_thresholds": tuned_metrics,
        "checkpoint": str(checkpoint_path),
        "thresholds": str(thresholds_path),
        "history": str(history_path),
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
