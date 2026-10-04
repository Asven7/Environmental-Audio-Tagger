#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from esaudio.config import load_config
from esaudio.manifests import assert_no_source_leakage, build_urbansound8k_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe UrbanSound8K manifests")
    parser.add_argument("--dataset-root", required=True, help="Directory containing audio/ and metadata/")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir", default="artifacts/manifests")
    parser.add_argument(
        "--include-mixtures",
        action="store_true",
        help="Also create configured synthetic mixture rows. Leave this off during Phase 4; mixtures are validated in Phase 5.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = build_urbansound8k_manifests(
        args.dataset_root,
        args.output_dir,
        config,
        include_mixtures=args.include_mixtures,
    )
    known_frames = {
        split: pd.read_csv(paths[f"known_{split}"])
        for split in ("train", "val", "test")
    }
    assert_no_source_leakage(known_frames)
    print("Created manifests:")
    for name, path in paths.items():
        print(f"  {name}: {Path(path)}")
    print("Source-group leakage check: PASSED")
    print(f"Mixtures included: {args.include_mixtures}")


if __name__ == "__main__":
    main()
