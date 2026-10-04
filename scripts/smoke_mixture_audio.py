#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from esaudio.audio import load_audio, mix_two_sources, pad_or_crop
from esaudio.config import load_config, window_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize a few real UrbanSound8K mixture rows without training"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-dir", default="artifacts/manifests_phase5")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--count", type=int, default=3)
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

    manifest_path = Path(args.manifest_dir) / f"known_{args.split}.csv"
    frame = pd.read_csv(manifest_path)
    mixtures = frame[frame["sample_type"].astype(str) == "mix"].head(args.count)
    if mixtures.empty:
        raise ValueError(f"No mixture rows found in {manifest_path}")

    dataset_root = Path(args.dataset_root)
    print("=== Real Mixture Audio Smoke Test ===")
    print(f"split={args.split} sample_rate={sample_rate} target_samples={target_samples}")
    print(f"crop_strategy={crop_strategy}")

    for _, row in mixtures.iterrows():
        source_a_path = dataset_root / str(row["source_a"])
        source_b_path = dataset_root / str(row["source_b"])
        a = pad_or_crop(load_audio(source_a_path, sample_rate), target_samples, crop_strategy)
        b = pad_or_crop(load_audio(source_b_path, sample_rate), target_samples, crop_strategy)
        mixed = mix_two_sources(
            a,
            b,
            relative_db=float(row["relative_db"]),
            overlap_ratio=float(row["overlap_ratio"]),
            target_dbfs=float(data_cfg.get("target_rms_dbfs", -20.0)),
        )
        if mixed.shape != (target_samples,):
            raise RuntimeError(f"Unexpected mixture shape: {mixed.shape}")
        if not np.isfinite(mixed).all():
            raise RuntimeError("Mixture contains non-finite samples")
        peak = float(np.max(np.abs(mixed)))
        if peak > 0.991:
            raise RuntimeError(f"Mixture peak exceeds safety limit: {peak}")
        print(
            f"PASS {row['sample_id']}: relative_db={float(row['relative_db']):g} "
            f"overlap={float(row['overlap_ratio']):g} peak={peak:.4f}"
        )

    print("Real mixture audio smoke test: PASSED")
    print("No files were written and no model training was performed.")


if __name__ == "__main__":
    main()
