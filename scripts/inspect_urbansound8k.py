#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from esaudio.config import load_config
from esaudio.manifests import audit_urbansound8k, load_urbansound8k_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit UrbanSound8K metadata and the configured experimental protocol without training."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default="artifacts/dataset_audit.json")
    args = parser.parse_args()

    config = load_config(args.config)
    report = audit_urbansound8k(args.dataset_root, config)
    metadata = load_urbansound8k_metadata(args.dataset_root)

    table = (
        metadata.groupby(["class", "fold"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(1, 11), fill_value=0)
        .sort_index()
    )
    table["TOTAL"] = table.sum(axis=1)
    table["SOURCE_RECORDINGS"] = metadata.groupby("class")["fsID"].nunique().reindex(table.index)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== UrbanSound8K Metadata Audit ===")
    print(f"Rows: {report['metadata_rows']}")
    print(f"Folds: {report['folds_present']}")
    print(
        f"Occurrence-group isolation check ({report['source_group_unit']}): "
        f"{report['source_group_leakage_check'].upper()}"
    )
    print(
        "fsIDs spanning multiple official folds (diagnostic, allowed): "
        f"{report['fsids_spanning_multiple_folds']} "
        f"examples={report['fsids_spanning_multiple_folds_examples']}"
    )
    print(
        "fsIDs spanning configured train/val/test splits (diagnostic): "
        f"{report['fsids_spanning_configured_splits']} "
        f"examples={report['fsids_spanning_configured_splits_examples']}"
    )
    print(
        "Research-manifest policy for cross-split fsIDs: "
        f"{report['cross_split_recording_policy']}"
    )
    print(
        f"Excluded cross-split fsIDs: {report['excluded_cross_split_fsids']} "
        f"clips={report['excluded_cross_split_clips']}"
    )
    if report["cross_split_fsid_details"]:
        print("Cross-split fsID details:")
        for item in report["cross_split_fsid_details"]:
            print(
                f"  fsID={item['fsID']} clips={item['clips']} "
                f"folds={item['folds']} splits={item['splits']} classes={item['classes']}"
            )
    print("\nConfigured target classes:")
    for name in report["configured_target_classes"]:
        print(f"  - {name}")
    print("Configured held-out classes:")
    for name in report["configured_heldout_classes"]:
        print(f"  - {name}")
    print("\nClips by class and official fold:")
    print(table.to_string())
    print("\nConfigured split totals (raw metadata):")
    for split, info in report["split_counts"].items():
        print(
            f"  {split}: folds={info['folds']} clips={info['clips']} "
            f"source_recordings={info['source_recordings']}"
        )
    print("Configured split totals after cross-split recording exclusion:")
    for split, info in report["effective_split_counts_after_exclusion"].items():
        print(
            f"  {split}: folds={info['folds']} clips={info['clips']} "
            f"source_recordings={info['source_recordings']}"
        )
    print(f"\nAudit JSON: {output_path}")
    print("No training or mixture generation was performed.")


if __name__ == "__main__":
    main()
