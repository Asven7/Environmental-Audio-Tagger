from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoints import load_checkpoint, load_thresholds
from .dataset import AudioManifestDataset
from .evaluation import group_metrics, multilabel_metrics, rejection_metrics
from .features import LogMelExtractor
from .training import predict_dataset, resolve_device


def evaluate_checkpoint(
    config: dict,
    checkpoint_path: str | Path,
    thresholds_path: str | Path,
    known_manifest: str | Path,
    audio_root: str | Path,
    split: str = "test",
    ood_manifest: str | Path | None = None,
    device_name: str = "cpu",
) -> dict:
    device = resolve_device(device_name)
    model, payload = load_checkpoint(checkpoint_path, device=device)
    class_names = list(payload["class_names"])
    thresholds = np.asarray(load_thresholds(thresholds_path, expected_classes=class_names), dtype=np.float32)
    extractor = LogMelExtractor(**payload["feature_config"]).to(device)
    criterion = nn.BCEWithLogitsLoss()

    known_dataset = AudioManifestDataset(known_manifest, audio_root, config, split=split)
    known_loader = DataLoader(
        known_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    known_bundle = predict_dataset(model, extractor, known_loader, criterion, device)
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
        "known_loss": float(known_bundle.mean_loss),
        "n_known": int(len(known_dataset)),
    }

    if ood_manifest is not None and Path(ood_manifest).exists():
        ood_dataset = AudioManifestDataset(
            ood_manifest,
            audio_root,
            config,
            split=split,
            include_ood=True,
        )
        if len(ood_dataset):
            ood_loader = DataLoader(
                ood_dataset,
                batch_size=int(config["training"]["batch_size"]),
                shuffle=False,
                num_workers=0,
            )
            ood_bundle = predict_dataset(model, extractor, ood_loader, criterion, device)
            result["rejection"] = rejection_metrics(
                known_bundle.scores,
                ood_bundle.scores,
                thresholds,
            )
            result["n_ood"] = int(len(ood_dataset))
    return result


def save_evaluation(result: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
