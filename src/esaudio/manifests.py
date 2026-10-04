from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_URBANSOUND8K_COLUMNS = {
    "slice_file_name",
    "fsID",
    "fold",
    "classID",
    "class",
}

OFFICIAL_CLASS_ID_TO_NAME = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}

MANIFEST_COLUMNS = [
    "sample_id",
    "split",
    "sample_type",
    "source_a",
    "source_b",
    "source_group_a",
    "source_group_b",
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


def load_urbansound8k_metadata(dataset_root: str | Path) -> pd.DataFrame:
    """Load and validate the official UrbanSound8K metadata CSV.

    The function validates schema and the official classID/class mapping but does
    not modify, reshuffle, or relabel the dataset.
    """
    dataset_root = Path(dataset_root)
    metadata_path = dataset_root / "metadata" / "UrbanSound8K.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Expected UrbanSound8K metadata at {metadata_path}. "
            "The dataset root must contain metadata/UrbanSound8K.csv and audio/fold1..fold10."
        )

    frame = pd.read_csv(metadata_path)
    missing = REQUIRED_URBANSOUND8K_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"UrbanSound8K metadata is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("UrbanSound8K metadata is empty")

    # Normalize only the fields we use; keep original rows/order otherwise.
    frame = frame.copy()
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["classID"] = pd.to_numeric(frame["classID"], errors="raise").astype(int)
    frame["fsID"] = pd.to_numeric(frame["fsID"], errors="raise").astype(int)
    frame["class"] = frame["class"].astype(str)
    frame["slice_file_name"] = frame["slice_file_name"].astype(str)

    invalid_folds = sorted(set(frame["fold"]) - set(range(1, 11)))
    if invalid_folds:
        raise ValueError(f"Unexpected UrbanSound8K fold values: {invalid_folds}")

    invalid_class_ids = sorted(set(frame["classID"]) - set(OFFICIAL_CLASS_ID_TO_NAME))
    if invalid_class_ids:
        raise ValueError(f"Unexpected UrbanSound8K classID values: {invalid_class_ids}")

    bad_pairs = frame[
        frame.apply(
            lambda row: OFFICIAL_CLASS_ID_TO_NAME[int(row["classID"])] != str(row["class"]),
            axis=1,
        )
    ]
    if not bad_pairs.empty:
        examples = bad_pairs[["classID", "class"]].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"UrbanSound8K classID/class mapping mismatch: {examples}")

    duplicated_names = frame[frame["slice_file_name"].duplicated(keep=False)]
    if not duplicated_names.empty:
        examples = duplicated_names["slice_file_name"].head(5).tolist()
        raise ValueError(f"Duplicate slice_file_name values found: {examples}")

    return frame


def _parse_occurrence_group(row: pd.Series) -> str:
    """Return the annotated-event occurrence identity encoded by UrbanSound8K.

    UrbanSound8K filenames follow:
    ``fsID-classID-occurrenceID-sliceID.wav``. Multiple overlapping slices from
    the same annotated occurrence are highly correlated and must stay in the
    same experimental split. The broader ``fsID`` alone is *not* used as a
    hard grouping key because one Freesound recording may contain different
    annotated events/classes that the official dataset assigns to different
    folds.
    """
    name = str(row["slice_file_name"])
    parts = Path(name).stem.split("-")
    if len(parts) != 4:
        raise ValueError(
            "Unexpected UrbanSound8K filename format; expected "
            "fsID-classID-occurrenceID-sliceID.wav: " + name
        )
    try:
        file_fsid, file_class_id, occurrence_id, _slice_id = (int(x) for x in parts)
    except ValueError as exc:
        raise ValueError(f"Non-integer UrbanSound8K filename component: {name}") from exc

    if file_fsid != int(row["fsID"]):
        raise ValueError(
            f"Filename/metadata fsID mismatch for {name}: "
            f"filename={file_fsid}, metadata={int(row['fsID'])}"
        )
    if file_class_id != int(row["classID"]):
        raise ValueError(
            f"Filename/metadata classID mismatch for {name}: "
            f"filename={file_class_id}, metadata={int(row['classID'])}"
        )
    return f"occurrence:{file_fsid}:{file_class_id}:{occurrence_id}"


def validate_occurrence_source_groups(metadata: pd.DataFrame) -> None:
    """Ensure all slices of one annotated occurrence remain in one fold.

    ``fsID`` identifies the broader Freesound recording, but official
    UrbanSound8K metadata can legitimately assign different annotated events
    from one ``fsID`` to different folds. The stable hard-isolation unit for
    this project is therefore ``(fsID, classID, occurrenceID)``.
    """
    groups = metadata.apply(_parse_occurrence_group, axis=1)
    folds_per_group = pd.DataFrame(
        {"group": groups, "fold": metadata["fold"].to_numpy()}
    ).groupby("group")["fold"].nunique()
    leaked = folds_per_group[folds_per_group > 1]
    if not leaked.empty:
        examples = [str(x) for x in leaked.index[:5]]
        raise ValueError(
            "Annotated-occurrence leakage detected: the same "
            "(fsID, classID, occurrenceID) appears in multiple folds. "
            f"Examples: {examples}"
        )


def _fsid_diagnostics(metadata: pd.DataFrame, config: dict) -> dict:
    """Report broader recording reuse without treating it as corruption."""
    work = metadata[["fsID", "fold"]].copy()
    work["split"] = work["fold"].map(lambda fold: assign_split(int(fold), config))

    folds_per_fsid = work.groupby("fsID")["fold"].nunique()
    multi_fold_ids = [int(x) for x in folds_per_fsid[folds_per_fsid > 1].index]

    split_work = work[work["split"].notna()]
    splits_per_fsid = split_work.groupby("fsID")["split"].nunique()
    cross_split_ids = [int(x) for x in splits_per_fsid[splits_per_fsid > 1].index]

    return {
        "fsids_spanning_multiple_folds": len(multi_fold_ids),
        "fsids_spanning_multiple_folds_examples": multi_fold_ids[:10],
        "fsids_spanning_configured_splits": len(cross_split_ids),
        "fsids_spanning_configured_splits_examples": cross_split_ids[:10],
    }


def cross_split_fsids(metadata: pd.DataFrame, config: dict) -> list[int]:
    """Return broader Freesound recording IDs that span configured splits.

    UrbanSound8K can legitimately place different annotated occurrences from one
    ``fsID`` in different official folds. For this project we preserve the
    official fold labels but exclude any broader recording that crosses the
    configured train/val/test boundary from research manifests. This provides a
    stricter source-independence guarantee without reassigning official folds.
    """
    work = metadata[["fsID", "fold"]].copy()
    work["split"] = work["fold"].map(lambda fold: assign_split(int(fold), config))
    work = work[work["split"].notna()]
    splits_per_fsid = work.groupby("fsID")["split"].nunique()
    return sorted(int(x) for x in splits_per_fsid[splits_per_fsid > 1].index)


def cross_split_fsid_details(metadata: pd.DataFrame, config: dict) -> list[dict]:
    """Describe cross-split ``fsID`` recordings for audit/reporting."""
    ids = cross_split_fsids(metadata, config)
    details: list[dict] = []
    for fsid in ids:
        subset = metadata[metadata["fsID"] == fsid].copy()
        subset["split"] = subset["fold"].map(lambda fold: assign_split(int(fold), config))
        subset = subset[subset["split"].notna()]
        details.append(
            {
                "fsID": int(fsid),
                "clips": int(len(subset)),
                "folds": sorted(int(x) for x in subset["fold"].unique()),
                "splits": sorted(str(x) for x in subset["split"].unique()),
                "classes": sorted(str(x) for x in subset["class"].unique()),
            }
        )
    return details


def filter_cross_split_recordings(
    metadata: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, list[int]]:
    """Exclude recordings that span train/val/test from research manifests.

    We intentionally *drop* these recordings instead of moving their clips to a
    different fold. This leaves the official UrbanSound8K fold assignment
    untouched while guaranteeing that no broader Freesound recording is shared
    across the configured experimental splits.
    """
    excluded = cross_split_fsids(metadata, config)
    if not excluded:
        return metadata.copy(), []
    filtered = metadata[~metadata["fsID"].isin(excluded)].copy()
    return filtered, excluded


def validate_protocol_classes(metadata: pd.DataFrame, config: dict) -> None:
    available = set(metadata["class"].unique())
    target = list(config["project"]["target_classes"])
    heldout = list(config["project"]["heldout_classes"])
    requested = set(target) | set(heldout)
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Configured classes are not present in UrbanSound8K metadata: {missing}")
    if len(requested) != len(target) + len(heldout):
        raise ValueError("Target and held-out class sets must be disjoint")


def audit_urbansound8k(dataset_root: str | Path, config: dict) -> dict:
    """Return a reproducible metadata/protocol audit without touching audio files."""
    metadata = load_urbansound8k_metadata(dataset_root)
    validate_occurrence_source_groups(metadata)
    validate_protocol_classes(metadata, config)

    fold_counts = (
        metadata.groupby(["class", "fold"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(1, 11), fill_value=0)
        .sort_index()
    )
    class_counts = metadata.groupby("class").size().sort_index()
    source_counts = metadata.groupby("class")["fsID"].nunique().sort_index()

    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        folds = [int(x) for x in config["data"][f"{split}_folds"]]
        subset = metadata[metadata["fold"].isin(folds)]
        split_counts[split] = {
            "folds": folds,
            "clips": int(len(subset)),
            "source_recordings": int(subset["fsID"].nunique()),
        }

    diagnostics = _fsid_diagnostics(metadata, config)
    cross_details = cross_split_fsid_details(metadata, config)
    filtered_metadata, excluded_fsids = filter_cross_split_recordings(metadata, config)
    filtered_split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        folds = [int(x) for x in config["data"][f"{split}_folds"]]
        subset = filtered_metadata[filtered_metadata["fold"].isin(folds)]
        filtered_split_counts[split] = {
            "folds": folds,
            "clips": int(len(subset)),
            "source_recordings": int(subset["fsID"].nunique()),
        }

    return {
        "metadata_rows": int(len(metadata)),
        "folds_present": sorted(int(x) for x in metadata["fold"].unique()),
        "classes_present": sorted(str(x) for x in metadata["class"].unique()),
        "class_clip_counts": {str(k): int(v) for k, v in class_counts.items()},
        "class_source_recording_counts": {str(k): int(v) for k, v in source_counts.items()},
        "class_fold_counts": {
            str(class_name): {str(int(fold)): int(value) for fold, value in row.items()}
            for class_name, row in fold_counts.iterrows()
        },
        "configured_target_classes": list(config["project"]["target_classes"]),
        "configured_heldout_classes": list(config["project"]["heldout_classes"]),
        "split_counts": split_counts,
        "source_group_unit": "fsID:classID:occurrenceID",
        "source_group_leakage_check": "passed",
        "cross_split_recording_policy": "exclude_from_all_research_manifests",
        "excluded_cross_split_fsids": excluded_fsids,
        "excluded_cross_split_clips": int(len(metadata) - len(filtered_metadata)),
        "cross_split_fsid_details": cross_details,
        "effective_split_counts_after_exclusion": filtered_split_counts,
        **diagnostics,
    }


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
        source_group = _parse_occurrence_group(row)
        base = {
            "sample_id": f"urbansound-{int(row['fsID'])}-{row['slice_file_name']}",
            "split": split,
            "sample_type": "single",
            "source_a": relative_path,
            "source_b": "",
            "source_group_a": source_group,
            "source_group_b": "",
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
    """Generate deterministic mixture metadata.

    The actual waveform mixing is performed later by the Dataset layer. This
    function never combines sources from different splits.
    """
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
        group_a_value = a.get("source_group_a", "")
        group_b_value = b.get("source_group_a", "")
        group_a = (
            str(group_a_value).strip()
            if pd.notna(group_a_value) and str(group_a_value).strip() and str(group_a_value).lower() != "nan"
            else f"path:{a['source_a']}"
        )
        group_b = (
            str(group_b_value).strip()
            if pd.notna(group_b_value) and str(group_b_value).strip() and str(group_b_value).lower() != "nan"
            else f"path:{b['source_a']}"
        )
        rows.append(
            {
                "sample_id": f"mix-{split}-{len(rows):06d}",
                "split": split,
                "sample_type": "mix",
                "source_a": str(a["source_a"]),
                "source_b": str(b["source_a"]),
                "source_group_a": group_a,
                "source_group_b": group_b,
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
    *,
    include_mixtures: bool = True,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mix_cfg = config["mixing"]
    target_classes = list(config["project"]["target_classes"])
    counts = {
        "train": int(mix_cfg.get("train_mixtures", 0)) if include_mixtures else 0,
        "val": int(mix_cfg.get("val_mixtures", 0)) if include_mixtures else 0,
        "test": int(mix_cfg.get("test_mixtures", 0)) if include_mixtures else 0,
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
                "include_mixtures": bool(include_mixtures),
                "splits": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    paths["summary"] = summary_path
    return paths


def build_urbansound8k_manifests(
    dataset_root: str | Path,
    output_dir: str | Path,
    config: dict,
    *,
    include_mixtures: bool = True,
) -> dict[str, Path]:
    metadata = load_urbansound8k_metadata(dataset_root)
    validate_occurrence_source_groups(metadata)
    validate_protocol_classes(metadata, config)
    filtered_metadata, excluded_fsids = filter_cross_split_recordings(metadata, config)
    known, ood = _make_single_rows(filtered_metadata, config)
    manifests = write_manifests(
        known,
        ood,
        output_dir,
        config,
        include_mixtures=include_mixtures,
    )

    summary_path = manifests["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_group_unit"] = "fsID:classID:occurrenceID"
    summary["cross_split_recording_policy"] = "exclude_from_all_research_manifests"
    summary["excluded_cross_split_fsids"] = excluded_fsids
    summary["excluded_cross_split_clips"] = int(len(metadata) - len(filtered_metadata))
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Final defense-in-depth check after manifest construction.
    known_frames = {
        split: pd.read_csv(manifests[f"known_{split}"])
        for split in ("train", "val", "test")
    }
    assert_no_source_leakage(known_frames)
    return manifests


def _source_keys(frame: pd.DataFrame) -> set[str]:
    """Return source-group identifiers with path fallback for legacy/demo rows."""
    keys: set[str] = set()
    for _, row in frame.iterrows():
        for suffix in ("a", "b"):
            source_column = f"source_{suffix}"
            group_column = f"source_group_{suffix}"
            source = str(row.get(source_column, "") or "").strip()
            group_value = row.get(group_column, "")
            group = ""
            if pd.notna(group_value):
                group = str(group_value).strip()
                if group.lower() == "nan":
                    group = ""
            if group:
                keys.add(group)
            elif source and source.lower() != "nan":
                keys.add(f"path:{source}")
    return keys


def assert_no_source_leakage(manifests: dict[str, pd.DataFrame]) -> None:
    """Raise if an original source group appears in more than one split."""
    split_sources = {split: _source_keys(frame) for split, frame in manifests.items()}
    splits = list(split_sources)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = split_sources[left] & split_sources[right]
            if overlap:
                examples = sorted(overlap)[:5]
                raise ValueError(
                    f"Source leakage detected between {left} and {right}: {examples}"
                )
