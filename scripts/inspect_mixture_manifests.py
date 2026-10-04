#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from esaudio.config import load_config
from esaudio.manifests import assert_no_source_leakage, validate_mixture_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit controlled multi-label mixture manifests without loading audio"
    )
    parser.add_argument("--manifest-dir", default="artifacts/manifests_phase5")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    config = load_config(args.config)
    target_classes = list(config["project"]["target_classes"])
    mix_cfg = config["mixing"]
    db_levels = [float(x) for x in mix_cfg["relative_db_levels"]]
    overlaps = [float(x) for x in mix_cfg["overlap_ratios"]]
    expected_counts = {
        "train": int(mix_cfg.get("train_mixtures", 0)),
        "val": int(mix_cfg.get("val_mixtures", 0)),
        "test": int(mix_cfg.get("test_mixtures", 0)),
    }

    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        path = manifest_dir / f"known_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing manifest: {path}")
        frames[split] = pd.read_csv(path)

    assert_no_source_leakage(frames)

    print("=== Controlled Mixture Manifest Audit ===")
    print(f"Manifest directory: {manifest_dir}")
    print(f"Target classes: {len(target_classes)}")
    print(f"Relative dB levels: {db_levels}")
    print(f"Overlap ratios: {overlaps}")
    print("Source-group leakage across splits: PASSED")

    for split in ("train", "val", "test"):
        frame = frames[split]
        singles = frame[frame["sample_type"].astype(str) == "single"].copy()
        mixtures = frame[frame["sample_type"].astype(str) == "mix"].copy()
        expected = expected_counts[split]
        if len(mixtures) != expected:
            raise ValueError(
                f"Expected {expected} mixtures for {split}, found {len(mixtures)}"
            )
        audit = validate_mixture_rows(
            mixtures,
            target_classes,
            db_levels,
            overlaps,
            expected_split=split,
            source_singles=singles,
            enforce_balance=True,
        )

        print(f"\n[{split}]")
        print(f"  known singles: {len(singles)}")
        print(f"  controlled mixtures: {len(mixtures)}")
        print(
            "  class-pair count range: "
            f"{audit['class_pair_count_min']}..{audit['class_pair_count_max']}"
        )
        print(
            "  dB/overlap condition count range: "
            f"{audit['condition_count_min']}..{audit['condition_count_max']}"
        )
        print(f"  unique source-group pairs: {audit['unique_source_group_pairs']}")
        print(f"  maximum source-group reuse: {audit['source_group_reuse_max']}")
        print("  condition matrix (rows=dB, columns=overlap):")
        matrix = (
            mixtures.groupby(["relative_db", "overlap_ratio"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=db_levels, columns=overlaps, fill_value=0)
        )
        print(matrix.to_string())

    for split in ("train", "val", "test"):
        ood_path = manifest_dir / f"ood_{split}.csv"
        if ood_path.exists():
            ood = pd.read_csv(ood_path)
            if not ood.empty and (ood["sample_type"].astype(str) == "mix").any():
                raise ValueError(f"OOD manifest unexpectedly contains mixtures: {ood_path}")

    print("\nControlled mixture protocol validation: PASSED")
    print("No audio loading or model training was performed.")


if __name__ == "__main__":
    main()
