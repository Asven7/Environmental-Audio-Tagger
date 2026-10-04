#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from esaudio.audio import load_audio, mix_two_sources, normalize_rms, pad_or_crop
from esaudio.config import load_config, window_samples
from esaudio.features import feature_extractor_from_config


def _single_waveform(
    dataset_root: Path,
    source: str,
    sample_rate: int,
    target_samples: int,
    crop_strategy: str,
    target_dbfs: float,
) -> np.ndarray:
    waveform = load_audio(dataset_root / source, sample_rate)
    waveform = pad_or_crop(waveform, target_samples, crop_strategy)
    return normalize_rms(waveform, target_dbfs)


def _mixture_waveform(
    dataset_root: Path,
    row: pd.Series,
    sample_rate: int,
    target_samples: int,
    crop_strategy: str,
    target_dbfs: float,
) -> np.ndarray:
    a = pad_or_crop(
        load_audio(dataset_root / str(row["source_a"]), sample_rate),
        target_samples,
        crop_strategy,
    )
    b = pad_or_crop(
        load_audio(dataset_root / str(row["source_b"]), sample_rate),
        target_samples,
        crop_strategy,
    )
    return mix_two_sources(
        a,
        b,
        relative_db=float(row["relative_db"]),
        overlap_ratio=float(row["overlap_ratio"]),
        target_dbfs=target_dbfs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Log-Mel extraction on real UrbanSound8K singles and mixtures"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-dir", default="artifacts/manifests_phase5")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--count", type=int, default=2, help="Singles and mixtures to sample")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    config = load_config(args.config)
    sample_rate = int(config["project"]["sample_rate"])
    target_samples = window_samples(config)
    data_cfg = config.get("data", {})
    crop_strategy = (
        str(data_cfg.get("train_crop_strategy", "energy"))
        if args.split == "train"
        else str(data_cfg.get("eval_crop_strategy", "center"))
    )
    target_dbfs = float(data_cfg.get("target_rms_dbfs", -20.0))

    manifest_path = Path(args.manifest_dir) / f"known_{args.split}.csv"
    frame = pd.read_csv(manifest_path)
    singles = frame[frame["sample_type"].astype(str) == "single"].head(args.count)
    mixtures = frame[frame["sample_type"].astype(str) == "mix"].head(args.count)
    if singles.empty or mixtures.empty:
        raise ValueError(f"Need both single and mix rows in {manifest_path}")

    dataset_root = Path(args.dataset_root)
    waveforms: list[np.ndarray] = []
    labels: list[str] = []
    for _, row in singles.iterrows():
        waveforms.append(
            _single_waveform(
                dataset_root,
                str(row["source_a"]),
                sample_rate,
                target_samples,
                crop_strategy,
                target_dbfs,
            )
        )
        labels.append(str(row["sample_id"]))
    for _, row in mixtures.iterrows():
        waveforms.append(
            _mixture_waveform(
                dataset_root,
                row,
                sample_rate,
                target_samples,
                crop_strategy,
                target_dbfs,
            )
        )
        labels.append(str(row["sample_id"]))

    batch = torch.from_numpy(np.stack(waveforms).astype(np.float32, copy=False))
    extractor = feature_extractor_from_config(config).eval()
    expected_frames = extractor.expected_num_frames(target_samples)

    with torch.inference_mode():
        cpu_features = extractor(batch)
        cpu_repeat = extractor(batch)

    expected_shape = (len(waveforms), 1, extractor.n_mels, expected_frames)
    if tuple(cpu_features.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected feature shape: {tuple(cpu_features.shape)} != {expected_shape}"
        )
    if not torch.isfinite(cpu_features).all():
        raise RuntimeError("CPU Log-Mel features contain non-finite values")
    torch.testing.assert_close(cpu_features, cpu_repeat, rtol=0.0, atol=0.0)

    print("=== Real Log-Mel Feature Smoke Test ===")
    print(
        f"split={args.split} sample_rate={sample_rate} samples={target_samples} "
        f"n_fft={extractor.n_fft} hop={extractor.hop_length} n_mels={extractor.n_mels}"
    )
    print(
        f"center={extractor.center} power={extractor.power:g} "
        f"log_floor={extractor.log_floor:g} normalize={extractor.normalize_features}"
    )
    print(f"feature_shape={tuple(cpu_features.shape)} expected_frames={expected_frames}")
    for index, sample_id in enumerate(labels):
        item = cpu_features[index]
        print(
            f"PASS {sample_id}: mean={float(item.mean()):.6f} "
            f"std={float(item.std()):.6f} min={float(item.min()):.4f} "
            f"max={float(item.max()):.4f}"
        )
    print("CPU determinism: PASSED")

    if torch.cuda.is_available():
        gpu_extractor = type(extractor)(**extractor.export_config()).cuda().eval()
        with torch.inference_mode():
            gpu_features = gpu_extractor(batch.cuda()).cpu()
        if tuple(gpu_features.shape) != expected_shape or not torch.isfinite(gpu_features).all():
            raise RuntimeError("CUDA Log-Mel feature sanity check failed")
        max_abs_diff = float((cpu_features - gpu_features).abs().max())
        if max_abs_diff > 5e-2:
            raise RuntimeError(
                f"CPU/CUDA Log-Mel difference is unexpectedly large: {max_abs_diff:.6f}"
            )
        print(f"CUDA feature shape/finite check: PASSED (max_abs_diff={max_abs_diff:.6f})")
    else:
        print("CUDA feature sanity: SKIPPED (CUDA unavailable)")

    print("Real Log-Mel feature smoke test: PASSED")
    print("No model training was performed and no feature files were written.")


if __name__ == "__main__":
    main()
