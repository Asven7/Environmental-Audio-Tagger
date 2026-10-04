from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
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


def _manifest_source_group(row: pd.Series, suffix: str) -> str:
    """Return a stable source-group key with a path fallback."""
    group_value = row.get(f"source_group_{suffix}", "")
    if pd.notna(group_value):
        group = str(group_value).strip()
        if group and group.lower() != "nan":
            return group
    source_value = row.get(f"source_{suffix}", "")
    source = "" if pd.isna(source_value) else str(source_value).strip()
    if source and source.lower() != "nan":
        return f"path:{source}"
    return ""


def _broad_fsid_from_manifest_source(row: pd.Series, suffix: str) -> int | None:
    """Best-effort extraction of the broader UrbanSound8K ``fsID``.

    Mixture generation uses this only as an additional within-split safeguard:
    if both source recordings are identifiable and share the same ``fsID``, the
    pair is rejected. Legacy/demo manifests that do not encode an UrbanSound8K
    identifier remain supported.
    """
    group = _manifest_source_group(row, suffix)
    if group.startswith("occurrence:"):
        parts = group.split(":")
        if len(parts) >= 4:
            try:
                return int(parts[1])
            except ValueError:
                pass

    source_value = row.get(f"source_{suffix}", "")
    source = "" if pd.isna(source_value) else str(source_value).strip()
    stem = Path(source).stem
    parts = stem.split("-")
    if len(parts) == 4:
        try:
            return int(parts[0])
        except ValueError:
            pass
    return None


def _single_label_from_row(row: pd.Series, target_classes: list[str]) -> tuple[int, str]:
    indices = parse_json_list(row.get("label_indices"))
    names = parse_json_list(row.get("label_names"))
    if len(indices) != 1 or len(names) != 1:
        raise ValueError("Mixture source rows must be single-label known samples")
    index = int(indices[0])
    name = str(names[0])
    if index < 0 or index >= len(target_classes) or target_classes[index] != name:
        raise ValueError(
            f"Single-source label metadata is not aligned: index={index}, name={name!r}"
        )
    return index, name


def _known_single_subset(
    singles: pd.DataFrame,
    target_classes: list[str],
    split: str,
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    subset = singles[singles["split"].astype(str) == str(split)].copy().reset_index(drop=True)
    if "sample_type" in subset.columns:
        subset = subset[subset["sample_type"].astype(str) == "single"].reset_index(drop=True)
    if "is_ood" in subset.columns:
        # In-memory manifests use bool; CSV reloads also infer True/False as bool.
        subset = subset[~subset["is_ood"].astype(bool)].reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"Cannot generate mixtures: no known singles for split={split}")

    by_class: dict[str, list[int]] = {name: [] for name in target_classes}
    for idx, row in subset.iterrows():
        _label_index, label_name = _single_label_from_row(row, target_classes)
        by_class[label_name].append(int(idx))

    missing = [name for name, rows in by_class.items() if not rows]
    if missing:
        raise ValueError(
            f"Cannot create a balanced mixture protocol for split={split}; "
            f"missing target classes: {missing}"
        )
    return subset, by_class


def _choose_balanced_key(counter: Counter, keys: list, tie_rank: dict) -> object:
    minimum = min(counter[key] for key in keys)
    candidates = [key for key in keys if counter[key] == minimum]
    return min(candidates, key=lambda key: tie_rank[key])


def _requires_unique_source_pairs(source_singles: pd.DataFrame) -> bool:
    """Return True for research manifests with explicit occurrence identities.

    UrbanSound8K research rows created in Phase 4 carry ``occurrence:...``
    source-group identifiers.  Tiny synthetic/demo manifests predate that schema
    and fall back to ``path:...`` identities.  Demo datasets can contain fewer
    possible source pairs than the configured smoke-test mixture count, so pair
    reuse is allowed only for that path-fallback case.  Research occurrence
    manifests remain strict.
    """
    if source_singles.empty:
        return True
    groups = [
        _manifest_source_group(row, "a")
        for _, row in source_singles.iterrows()
    ]
    return bool(groups) and all(str(group).startswith("occurrence:") for group in groups)


def _select_source_pair(
    subset: pd.DataFrame,
    by_class: dict[str, list[int]],
    label_a: str,
    label_b: str,
    rng: np.random.Generator,
    used_group_pairs: set[tuple[str, str]],
    *,
    require_unique_pair: bool = True,
    max_attempts: int = 5000,
) -> tuple[pd.Series, pd.Series, str, str]:
    """Choose a source pair, avoiding same-recording self-mixtures.

    Research occurrence manifests require unique source-group pairs.  Tiny
    synthetic/demo path-fallback manifests may reuse pairs when their finite
    combinatorial capacity is smaller than the configured smoke-test count.
    """
    rows_a = by_class[label_a]
    rows_b = by_class[label_b]
    for _ in range(max_attempts):
        a = subset.iloc[int(rows_a[int(rng.integers(0, len(rows_a)))])]
        b = subset.iloc[int(rows_b[int(rng.integers(0, len(rows_b)))])]
        group_a = _manifest_source_group(a, "a")
        group_b = _manifest_source_group(b, "a")
        if not group_a or not group_b or group_a == group_b:
            continue
        pair_key = tuple(sorted((group_a, group_b)))
        if require_unique_pair and pair_key in used_group_pairs:
            continue

        fsid_a = _broad_fsid_from_manifest_source(a, "a")
        fsid_b = _broad_fsid_from_manifest_source(b, "a")
        if fsid_a is not None and fsid_b is not None and fsid_a == fsid_b:
            continue

        if require_unique_pair:
            used_group_pairs.add(pair_key)
        return a, b, group_a, group_b

    qualifier = "unique " if require_unique_pair else ""
    raise RuntimeError(
        f"Could not find a {qualifier}source pair for classes {label_a!r}/{label_b!r}. "
        "The requested mixture count may exceed the available compatible source pairs."
    )


def validate_mixture_rows(
    mixtures: pd.DataFrame,
    target_classes: list[str],
    relative_db_levels: list[float],
    overlap_ratios: list[float],
    *,
    expected_split: str | None = None,
    source_singles: pd.DataFrame | None = None,
    enforce_balance: bool = True,
    require_unique_source_pairs: bool | None = None,
) -> dict:
    """Validate controlled-mixture metadata and return compact audit statistics.

    The checks intentionally operate on metadata rather than audio waveforms.
    Waveform-level dB/overlap/clipping behavior is tested separately in
    ``tests/test_audio.py`` and ``tests/test_mixture_protocol.py``.
    """
    if mixtures.empty:
        return {
            "rows": 0,
            "class_pair_count_min": 0,
            "class_pair_count_max": 0,
            "condition_count_min": 0,
            "condition_count_max": 0,
            "unique_source_group_pairs": 0,
            "unique_source_pairs_required": bool(
                True if require_unique_source_pairs is None else require_unique_source_pairs
            ),
            "source_group_reuse_max": 0,
            "class_pair_counts": {},
            "condition_counts": {},
        }

    required = set(MANIFEST_COLUMNS)
    missing_columns = required - set(mixtures.columns)
    if missing_columns:
        raise ValueError(f"Mixture manifest is missing columns: {sorted(missing_columns)}")
    if mixtures["sample_id"].astype(str).duplicated().any():
        raise ValueError("Mixture sample_id values must be unique")
    if not (mixtures["sample_type"].astype(str) == "mix").all():
        raise ValueError("validate_mixture_rows expects only sample_type='mix' rows")
    if mixtures["is_ood"].astype(bool).any():
        raise ValueError("Known-class mixtures may not be marked as OOD")
    if expected_split is not None and not (
        mixtures["split"].astype(str) == str(expected_split)
    ).all():
        raise ValueError(f"Mixture rows contain a split other than {expected_split!r}")

    allowed_db = [float(x) for x in relative_db_levels]
    allowed_overlap = [float(x) for x in overlap_ratios]
    if not allowed_db or not allowed_overlap:
        raise ValueError("Mixture dB levels and overlap ratios must be non-empty")

    class_pairs = list(combinations(target_classes, 2))
    pair_counts: Counter = Counter({pair: 0 for pair in class_pairs})
    conditions = [(db, overlap) for db in allowed_db for overlap in allowed_overlap]
    condition_counts: Counter = Counter({condition: 0 for condition in conditions})
    seen_group_pairs: set[tuple[str, str]] = set()
    source_group_reuse: Counter = Counter()

    single_source_labels: dict[tuple[str, str], tuple[int, str]] = {}
    if source_singles is not None:
        singles = source_singles.copy()
        if "sample_type" in singles.columns:
            singles = singles[singles["sample_type"].astype(str) == "single"]
        for _, single in singles.iterrows():
            idx, name = _single_label_from_row(single, target_classes)
            source = str(single.get("source_a", "")).strip()
            group = _manifest_source_group(single, "a")
            single_source_labels[(source, group)] = (idx, name)

    if require_unique_source_pairs is None:
        require_unique_source_pairs = (
            True if source_singles is None else _requires_unique_source_pairs(source_singles)
        )

    class_index = {name: idx for idx, name in enumerate(target_classes)}
    orientation_counts: dict[tuple[str, str], Counter] = {
        pair: Counter() for pair in class_pairs
    }

    for _, row in mixtures.iterrows():
        source_a = str(row["source_a"]).strip()
        source_b = str(row["source_b"]).strip()
        if not source_a or not source_b or source_a == source_b:
            raise ValueError("Each mixture must reference two distinct non-empty source paths")

        group_a = _manifest_source_group(row, "a")
        group_b = _manifest_source_group(row, "b")
        if not group_a or not group_b or group_a == group_b:
            raise ValueError("Each mixture must reference two distinct source groups")
        group_pair = tuple(sorted((group_a, group_b)))
        if require_unique_source_pairs and group_pair in seen_group_pairs:
            raise ValueError(f"Duplicate source-group pair in mixtures: {group_pair}")
        seen_group_pairs.add(group_pair)
        source_group_reuse[group_a] += 1
        source_group_reuse[group_b] += 1

        fsid_a = _broad_fsid_from_manifest_source(row, "a")
        fsid_b = _broad_fsid_from_manifest_source(row, "b")
        if fsid_a is not None and fsid_b is not None and fsid_a == fsid_b:
            raise ValueError(f"Mixture uses two events from the same broader fsID={fsid_a}")

        indices = [int(x) for x in parse_json_list(row["label_indices"])]
        names = [str(x) for x in parse_json_list(row["label_names"])]
        if len(indices) != 2 or len(set(indices)) != 2 or len(names) != 2:
            raise ValueError("Every controlled mixture must have exactly two distinct known labels")
        if indices != sorted(indices):
            raise ValueError("Mixture label_indices must be sorted in target-class order")
        expected_names = [target_classes[idx] for idx in indices]
        if names != expected_names:
            raise ValueError(
                f"Mixture label_names are not aligned with label_indices: {names} vs {expected_names}"
            )
        canonical_pair = tuple(expected_names)
        if canonical_pair not in pair_counts:
            raise ValueError(f"Unexpected class pair: {canonical_pair}")
        pair_counts[canonical_pair] += 1

        db = float(row["relative_db"])
        overlap = float(row["overlap_ratio"])
        db_match = next((x for x in allowed_db if np.isclose(db, x)), None)
        overlap_match = next((x for x in allowed_overlap if np.isclose(overlap, x)), None)
        if db_match is None:
            raise ValueError(f"Unexpected relative_db={db}")
        if overlap_match is None:
            raise ValueError(f"Unexpected overlap_ratio={overlap}")
        condition_counts[(float(db_match), float(overlap_match))] += 1

        if source_singles is not None:
            key_a = (source_a, group_a)
            key_b = (source_b, group_b)
            if key_a not in single_source_labels or key_b not in single_source_labels:
                raise ValueError("A mixture source is not present among the same-split known singles")
            idx_a, name_a = single_source_labels[key_a]
            idx_b, name_b = single_source_labels[key_b]
            if name_a == name_b:
                raise ValueError("Mixture sources must come from different classes")
            if sorted((idx_a, idx_b)) != indices:
                raise ValueError("Mixture target labels do not equal the union of source labels")
            orientation_counts[canonical_pair][name_a] += 1

    if enforce_balance:
        pair_values = list(pair_counts.values())
        if max(pair_values) - min(pair_values) > 1:
            raise ValueError("Class-pair mixture counts are not balanced within one sample")
        condition_values = list(condition_counts.values())
        if max(condition_values) - min(condition_values) > 1:
            raise ValueError("dB/overlap condition counts are not balanced within one sample")
        if source_singles is not None:
            for pair, counts in orientation_counts.items():
                if pair_counts[pair] == 0:
                    continue
                left = counts[pair[0]]
                right = counts[pair[1]]
                if abs(left - right) > 1:
                    raise ValueError(
                        f"Source A/B orientation is imbalanced for class pair {pair}: {left} vs {right}"
                    )

    pair_counts_json = {"|".join(pair): int(pair_counts[pair]) for pair in class_pairs}
    condition_counts_json = {
        f"db={db:g}|overlap={overlap:g}": int(condition_counts[(db, overlap)])
        for db, overlap in conditions
    }
    pair_values = list(pair_counts.values())
    condition_values = list(condition_counts.values())
    return {
        "rows": int(len(mixtures)),
        "class_pair_count_min": int(min(pair_values)),
        "class_pair_count_max": int(max(pair_values)),
        "condition_count_min": int(min(condition_values)),
        "condition_count_max": int(max(condition_values)),
        "unique_source_group_pairs": int(len(seen_group_pairs)),
        "unique_source_pairs_required": bool(require_unique_source_pairs),
        "source_group_reuse_max": int(max(source_group_reuse.values(), default=0)),
        "class_pair_counts": pair_counts_json,
        "condition_counts": condition_counts_json,
    }


def generate_mixture_rows(
    singles: pd.DataFrame,
    target_classes: list[str],
    split: str,
    count: int,
    relative_db_levels: list[float],
    overlap_ratios: list[float],
    seed: int,
) -> pd.DataFrame:
    """Generate deterministic, balanced, leakage-safe mixture metadata.

    The generator balances unordered class-pair counts and global experimental
    condition counts (relative dB × temporal overlap) to within one sample. It
    also balances which class occupies source A within each pair, preventing the
    sign of ``relative_db`` from becoming systematically associated with one
    class. Exact source-group pairs are never repeated, and two annotated events
    from the same broader UrbanSound8K ``fsID`` are not paired when that identity
    is available.

    Waveforms are mixed later by ``AudioManifestDataset`` using ``mix_two_sources``.
    """
    if count <= 0:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    if len(set(target_classes)) != len(target_classes) or len(target_classes) < 2:
        raise ValueError("target_classes must contain at least two unique classes")

    db_levels = [float(x) for x in relative_db_levels]
    overlaps = [float(x) for x in overlap_ratios]
    if not db_levels or not overlaps:
        raise ValueError("relative_db_levels and overlap_ratios must be non-empty")
    if not all(np.isfinite(x) for x in db_levels):
        raise ValueError("relative_db_levels must be finite")
    if not all(0.0 < x <= 1.0 for x in overlaps):
        raise ValueError("overlap_ratios must lie in (0, 1]")

    subset, by_class = _known_single_subset(singles, target_classes, split)
    require_unique_source_pairs = _requires_unique_source_pairs(subset)
    class_pairs = list(combinations(target_classes, 2))
    conditions = [(db, overlap) for db in db_levels for overlap in overlaps]
    rng = np.random.default_rng(int(seed))

    pair_rank_values = rng.permutation(len(class_pairs))
    pair_rank = {pair: int(rank) for rank, pair in enumerate(
        [class_pairs[int(i)] for i in pair_rank_values]
    )}
    condition_rank_values = rng.permutation(len(conditions))
    condition_rank = {condition: int(rank) for rank, condition in enumerate(
        [conditions[int(i)] for i in condition_rank_values]
    )}
    orientation_offsets = {
        pair: int(rng.integers(0, 2)) for pair in class_pairs
    }

    pair_counts: Counter = Counter({pair: 0 for pair in class_pairs})
    condition_counts: Counter = Counter({condition: 0 for condition in conditions})
    joint_counts: Counter = Counter()
    orientation_use: Counter = Counter({pair: 0 for pair in class_pairs})
    used_group_pairs: set[tuple[str, str]] = set()
    rows: list[dict] = []
    class_to_index = {name: index for index, name in enumerate(target_classes)}

    for row_index in range(int(count)):
        pair = _choose_balanced_key(pair_counts, class_pairs, pair_rank)
        # Among globally least-used conditions, prefer the condition least used
        # with this class pair. This reduces pair/condition confounding while
        # preserving exact global balance.
        min_condition_count = min(condition_counts[c] for c in conditions)
        condition_candidates = [c for c in conditions if condition_counts[c] == min_condition_count]
        condition = min(
            condition_candidates,
            key=lambda c: (joint_counts[(pair, c)], condition_rank[c]),
        )

        use_count = orientation_use[pair]
        if (use_count + orientation_offsets[pair]) % 2 == 0:
            label_a, label_b = pair
        else:
            label_a, label_b = pair[1], pair[0]

        a, b, group_a, group_b = _select_source_pair(
            subset,
            by_class,
            label_a,
            label_b,
            rng,
            used_group_pairs,
            require_unique_pair=require_unique_source_pairs,
        )
        idx_a = class_to_index[label_a]
        idx_b = class_to_index[label_b]
        sorted_indices = sorted((idx_a, idx_b))
        aligned_names = [target_classes[idx] for idx in sorted_indices]
        relative_db, overlap = condition

        rows.append(
            {
                "sample_id": f"mix-{split}-{row_index:06d}",
                "split": split,
                "sample_type": "mix",
                "source_a": str(a["source_a"]),
                "source_b": str(b["source_a"]),
                "source_group_a": group_a,
                "source_group_b": group_b,
                "label_indices": _json_list(sorted_indices),
                "label_names": _json_list(aligned_names),
                "relative_db": float(relative_db),
                "overlap_ratio": float(overlap),
                "is_ood": False,
            }
        )
        pair_counts[pair] += 1
        condition_counts[condition] += 1
        joint_counts[(pair, condition)] += 1
        orientation_use[pair] += 1

    mixtures = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    validate_mixture_rows(
        mixtures,
        target_classes,
        db_levels,
        overlaps,
        expected_split=split,
        source_singles=subset,
        enforce_balance=True,
        require_unique_source_pairs=require_unique_source_pairs,
    )
    return mixtures

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
    relative_db_levels = [float(x) for x in mix_cfg["relative_db_levels"]]
    overlap_ratios = [float(x) for x in mix_cfg["overlap_ratios"]]
    counts = {
        "train": int(mix_cfg.get("train_mixtures", 0)) if include_mixtures else 0,
        "val": int(mix_cfg.get("val_mixtures", 0)) if include_mixtures else 0,
        "test": int(mix_cfg.get("test_mixtures", 0)) if include_mixtures else 0,
    }
    base_seed = int(config["training"].get("data_seed", 1234))
    paths: dict[str, Path] = {}
    summary: dict[str, dict[str, int]] = {}
    mixture_audits: dict[str, dict] = {}

    for split_idx, split in enumerate(("train", "val", "test")):
        single_split = known_singles[known_singles["split"] == split].copy().reset_index(drop=True)
        mix_split = generate_mixture_rows(
            known_singles,
            target_classes,
            split,
            counts[split],
            relative_db_levels,
            overlap_ratios,
            base_seed + split_idx,
        )
        mixture_audits[split] = validate_mixture_rows(
            mix_split,
            target_classes,
            relative_db_levels,
            overlap_ratios,
            expected_split=split,
            source_singles=single_split,
            enforce_balance=True,
            require_unique_source_pairs=_requires_unique_source_pairs(single_split),
        )
        known = (
            single_split
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
                "mixture_protocol": {
                    "generator": "balanced_pair_condition_v1",
                    "data_seed": base_seed,
                    "relative_db_levels": relative_db_levels,
                    "overlap_ratios": overlap_ratios,
                    "different_class_only": True,
                    "duplicate_source_group_pairs_allowed": bool(
                        any(
                            not audit.get("unique_source_pairs_required", True)
                            for audit in mixture_audits.values()
                        )
                    ),
                    "research_occurrence_pairs_remain_unique": True,
                    "same_broad_fsid_pairing_allowed_when_identifiable": False,
                    "class_pair_balance_tolerance": 1,
                    "condition_balance_tolerance": 1,
                    "source_orientation_balance_tolerance": 1,
                    "split_audits": mixture_audits,
                },
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
