from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MANIFEST_COLUMNS = [
    "sample_id",
    "split",
    "sample_type",
    "source_a",
    "source_b",
    "label_indices",
    "label_names",
    "relative_db",
    "overlap_ratio",
    "is_ood",
]


@dataclass(frozen=True)
class SplitSummary:
    split: str
    known_singles: int
    known_mixtures: int
    ood_singles: int


def _json_list(values: Iterable) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def parse_json_list(value: str | list | tuple | None) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if not str(value).strip():
        return []
    return json.loads(str(value))


def assign_split(fold: int, config: dict) -> str | None:
    fold = int(fold)
    data_cfg = config["data"]
    if fold in [int(x) for x in data_cfg["train_folds"]]:
        return "train"
    if fold in [int(x) for x in data_cfg["val_folds"]]:
        return "val"
    if fold in [int(x) for x in data_cfg["test_folds"]]:
        return "test"
    return None


def _make_single_rows(
    metadata: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_classes = list(config["project"]["target_classes"])
    heldout_classes = list(config["project"]["heldout_classes"])
    class_to_index = {name: index for index, name in enumerate(target_classes)}

    known_rows: list[dict] = []
    ood_rows: list[dict] = []
    for _, row in metadata.iterrows():
        class_name = str(row["class"])
        fold = int(row["fold"])
        split = assign_split(fold, config)
        if split is None:
            continue
        relative_path = f"audio/fold{fold}/{row['slice_file_name']}"
        base = {
            "sample_id": f"urbansound-{row['fsID']}-{row['slice_file_name']}",
            "split": split,
            "sample_type": "single",
            "source_a": relative_path,
            "source_b": "",
            "relative_db": np.nan,
            "overlap_ratio": 0.0,
        }
        if class_name in class_to_index:
            known_rows.append(
                {
                    **base,
                    "label_indices": _json_list([class_to_index[class_name]]),
                    "label_names": _json_list([class_name]),
                    "is_ood": False,
                }
            )
        elif class_name in heldout_classes and split in {"val", "test"}:
            ood_rows.append(
                {
                    **base,
                    "label_indices": _json_list([]),
                    "label_names": _json_list([class_name]),
                    "is_ood": True,
                }
            )
    return pd.DataFrame(known_rows, columns=MANIFEST_COLUMNS), pd.DataFrame(ood_rows, columns=MANIFEST_COLUMNS)


def generate_mixture_rows(
    singles: pd.DataFrame,
    target_classes: list[str],
    split: str,
    count: int,
    relative_db_levels: list[float],
    overlap_ratios: list[float],
    seed: int,
) -> pd.DataFrame:
    if count <= 0:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    subset = singles[(singles["split"] == split) & (~singles["is_ood"].astype(bool))].reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"Cannot generate mixtures: no known singles for split={split}")

    labels = subset["label_names"].apply(lambda x: parse_json_list(x)[0]).to_numpy()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    attempts = 0
    max_attempts = max(count * 50, 1000)
    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, len(subset)))
        j = int(rng.integers(0, len(subset)))
        if i == j or labels[i] == labels[j]:
            continue
        a = subset.iloc[i]
        b = subset.iloc[j]
        name_a = labels[i]
        name_b = labels[j]
        idx_a = target_classes.index(name_a)
        idx_b = target_classes.index(name_b)
        relative_db = float(relative_db_levels[int(rng.integers(0, len(relative_db_levels)))])
        overlap = float(overlap_ratios[int(rng.integers(0, len(overlap_ratios)))])
        rows.append(
            {
                "sample_id": f"mix-{split}-{len(rows):06d}",
                "split": split,
                "sample_type": "mix",
                "source_a": str(a["source_a"]),
                "source_b": str(b["source_a"]),
                "label_indices": _json_list(sorted([idx_a, idx_b])),
                "label_names": _json_list(sorted([name_a, name_b])),
                "relative_db": relative_db,
                "overlap_ratio": overlap,
                "is_ood": False,
            }
        )
    if len(rows) != count:
        raise RuntimeError(f"Generated only {len(rows)}/{count} mixtures after {attempts} attempts")
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def write_manifests(
    known_singles: pd.DataFrame,
    ood_singles: pd.DataFrame,
    output_dir: str | Path,
    config: dict,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mix_cfg = config["mixing"]
    target_classes = list(config["project"]["target_classes"])
    counts = {
        "train": int(mix_cfg.get("train_mixtures", 0)),
        "val": int(mix_cfg.get("val_mixtures", 0)),
        "test": int(mix_cfg.get("test_mixtures", 0)),
    }
    base_seed = int(config["training"].get("data_seed", 1234))
    paths: dict[str, Path] = {}
    summary: dict[str, dict[str, int]] = {}

    for split_idx, split in enumerate(("train", "val", "test")):
        single_split = known_singles[known_singles["split"] == split].copy()
        mix_split = generate_mixture_rows(
            known_singles,
            target_classes,
            split,
            counts[split],
            [float(x) for x in mix_cfg["relative_db_levels"]],
            [float(x) for x in mix_cfg["overlap_ratios"]],
            base_seed + split_idx,
        )
        known = (
            single_split.reset_index(drop=True)
            if mix_split.empty
            else pd.concat([single_split, mix_split], ignore_index=True)
        )
        known_path = output_dir / f"known_{split}.csv"
        known.to_csv(known_path, index=False)
        paths[f"known_{split}"] = known_path

        ood_split = ood_singles[ood_singles["split"] == split].copy()
        ood_path = output_dir / f"ood_{split}.csv"
        ood_split.to_csv(ood_path, index=False)
        paths[f"ood_{split}"] = ood_path

        summary[split] = {
            "known_singles": int(len(single_split)),
            "known_mixtures": int(len(mix_split)),
            "ood_singles": int(len(ood_split)),
        }

    summary_path = output_dir / "manifest_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "target_classes": target_classes,
                "heldout_classes": list(config["project"]["heldout_classes"]),
                "splits": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    paths["summary"] = summary_path
    return paths


def build_urbansound8k_manifests(dataset_root: str | Path, output_dir: str | Path, config: dict) -> dict[str, Path]:
    dataset_root = Path(dataset_root)
    metadata_path = dataset_root / "metadata" / "UrbanSound8K.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Expected UrbanSound8K metadata at {metadata_path}. Download the dataset separately and pass its root directory."
        )
    metadata = pd.read_csv(metadata_path)
    required_columns = {"slice_file_name", "fsID", "fold", "class"}
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"UrbanSound8K metadata is missing columns: {sorted(missing)}")
    known, ood = _make_single_rows(metadata, config)
    return write_manifests(known, ood, output_dir, config)


def assert_no_source_leakage(manifests: dict[str, pd.DataFrame]) -> None:
    """Raise if any original source path appears in more than one known split."""
    split_sources: dict[str, set[str]] = {}
    for split, frame in manifests.items():
        sources: set[str] = set()
        for column in ("source_a", "source_b"):
            if column not in frame:
                continue
            for value in frame[column].fillna("").astype(str):
                if value.strip():
                    sources.add(value.strip())
        split_sources[split] = sources

    splits = list(split_sources)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = split_sources[left] & split_sources[right]
            if overlap:
                examples = sorted(overlap)[:5]
                raise ValueError(
                    f"Source leakage detected between {left} and {right}: {examples}"
                )
