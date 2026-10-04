from pathlib import Path

import pandas as pd
import pytest

from esaudio.manifests import (
    OFFICIAL_CLASS_ID_TO_NAME,
    assert_no_source_leakage,
    audit_urbansound8k,
    build_urbansound8k_manifests,
    cross_split_fsids,
    filter_cross_split_recordings,
    load_urbansound8k_metadata,
    validate_occurrence_source_groups,
)


def _config() -> dict:
    return {
        "project": {
            "target_classes": [
                "air_conditioner",
                "car_horn",
                "children_playing",
                "dog_bark",
                "drilling",
                "engine_idling",
                "jackhammer",
                "siren",
            ],
            "heldout_classes": ["gun_shot", "street_music"],
        },
        "data": {
            "train_folds": [1, 2, 3, 4, 5, 6, 7],
            "val_folds": [8],
            "test_folds": [9, 10],
        },
        "mixing": {
            "relative_db_levels": [-6.0, 0.0, 6.0],
            "overlap_ratios": [0.25, 0.5, 1.0],
            "train_mixtures": 4,
            "val_mixtures": 2,
            "test_mixtures": 2,
        },
        "training": {"data_seed": 1234},
    }


def _write_fake_dataset(root: Path) -> None:
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True)
    rows = []
    counter = 1000
    # One source recording per class/fold is enough for protocol tests.
    for fold in range(1, 11):
        for class_id, class_name in OFFICIAL_CLASS_ID_TO_NAME.items():
            counter += 1
            rows.append(
                {
                    "slice_file_name": f"{counter}-{class_id}-0-0.wav",
                    "fsID": counter,
                    "start": 0.0,
                    "end": 2.0,
                    "salience": 1,
                    "fold": fold,
                    "classID": class_id,
                    "class": class_name,
                }
            )
    pd.DataFrame(rows).to_csv(metadata_dir / "UrbanSound8K.csv", index=False)


def test_metadata_loader_validates_official_schema_and_mapping(tmp_path):
    _write_fake_dataset(tmp_path)
    frame = load_urbansound8k_metadata(tmp_path)
    assert len(frame) == 100
    assert set(frame["fold"]) == set(range(1, 11))
    assert set(frame["class"]) == set(OFFICIAL_CLASS_ID_TO_NAME.values())


def test_metadata_loader_rejects_bad_class_mapping(tmp_path):
    _write_fake_dataset(tmp_path)
    path = tmp_path / "metadata" / "UrbanSound8K.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "class"] = "wrong_name"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="classID/class mapping mismatch"):
        load_urbansound8k_metadata(tmp_path)


def test_occurrence_isolation_allows_fsid_reuse_but_rejects_same_occurrence_across_folds(tmp_path):
    _write_fake_dataset(tmp_path)
    frame = load_urbansound8k_metadata(tmp_path)

    # Official UrbanSound8K can reuse one broader fsID for a different annotated
    # class/occurrence in another fold. That is diagnostic, not corruption.
    different_event = frame.iloc[[0]].copy()
    different_event["fold"] = 2
    different_event["classID"] = 9
    different_event["class"] = "street_music"
    fsid = int(different_event.iloc[0]["fsID"] )
    different_event["slice_file_name"] = f"{fsid}-9-1-0.wav"
    allowed = pd.concat([frame, different_event], ignore_index=True)
    validate_occurrence_source_groups(allowed)
    allowed.to_csv(tmp_path / "metadata" / "UrbanSound8K.csv", index=False)
    report = audit_urbansound8k(tmp_path, _config())
    assert report["fsids_spanning_multiple_folds"] == 1

    # The exact same annotated occurrence may not be split across folds.
    duplicate_occurrence = frame.iloc[[0]].copy()
    duplicate_occurrence["fold"] = 2
    duplicate_occurrence["slice_file_name"] = duplicate_occurrence["slice_file_name"].str.replace(
        "-0-0.wav", "-0-1.wav", regex=False
    )
    leaked = pd.concat([frame, duplicate_occurrence], ignore_index=True)
    with pytest.raises(ValueError, match="Annotated-occurrence leakage detected"):
        validate_occurrence_source_groups(leaked)


def test_audit_reports_class_and_split_counts(tmp_path):
    _write_fake_dataset(tmp_path)
    report = audit_urbansound8k(tmp_path, _config())
    assert report["metadata_rows"] == 100
    assert report["source_group_leakage_check"] == "passed"
    assert report["source_group_unit"] == "fsID:classID:occurrenceID"
    assert report["class_clip_counts"]["dog_bark"] == 10
    assert report["split_counts"]["train"]["clips"] == 70
    assert report["split_counts"]["val"]["clips"] == 10
    assert report["split_counts"]["test"]["clips"] == 20


def test_build_singles_only_manifests_keeps_heldout_out_of_train(tmp_path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "manifests"
    _write_fake_dataset(dataset_root)
    paths = build_urbansound8k_manifests(
        dataset_root,
        output_dir,
        _config(),
        include_mixtures=False,
    )
    train = pd.read_csv(paths["known_train"])
    val = pd.read_csv(paths["known_val"])
    test = pd.read_csv(paths["known_test"])
    ood_train = pd.read_csv(paths["ood_train"])
    ood_val = pd.read_csv(paths["ood_val"])
    ood_test = pd.read_csv(paths["ood_test"])

    assert set(train["sample_type"]) == {"single"}
    assert len(train) == 7 * 8
    assert len(val) == 8
    assert len(test) == 2 * 8
    assert len(ood_train) == 0
    assert len(ood_val) == 2
    assert len(ood_test) == 4
    assert "source_group_a" in train.columns

    assert_no_source_leakage({"train": train, "val": val, "test": test})


def test_assert_no_source_leakage_uses_source_group_not_only_file_path():
    train = pd.DataFrame(
        [{"source_a": "audio/fold1/a.wav", "source_group_a": "fsID:42", "source_b": "", "source_group_b": ""}]
    )
    test = pd.DataFrame(
        [{"source_a": "audio/fold9/b.wav", "source_group_a": "fsID:42", "source_b": "", "source_group_b": ""}]
    )
    with pytest.raises(ValueError, match="Source leakage detected"):
        assert_no_source_leakage({"train": train, "test": test})


def test_leakage_check_falls_back_to_paths_for_legacy_blank_source_groups():
    train = pd.DataFrame(
        [{"source_a": "audio/train/a.wav", "source_group_a": float("nan"), "source_b": "", "source_group_b": ""}]
    )
    test = pd.DataFrame(
        [{"source_a": "audio/test/b.wav", "source_group_a": float("nan"), "source_b": "", "source_group_b": ""}]
    )
    assert_no_source_leakage({"train": train, "test": test})


def test_cross_split_fsid_filter_drops_recording_without_reassigning_folds(tmp_path):
    _write_fake_dataset(tmp_path)
    path = tmp_path / "metadata" / "UrbanSound8K.csv"
    frame = pd.read_csv(path)

    # Reuse one train fsID for a distinct class/occurrence in validation.
    train_row = frame[(frame["fold"] == 1) & (frame["class"] == "siren")].iloc[[0]].copy()
    fsid = int(train_row.iloc[0]["fsID"])
    val_row = frame[(frame["fold"] == 8) & (frame["class"] == "engine_idling")].iloc[[0]].copy()
    val_row["fsID"] = fsid
    val_row["slice_file_name"] = f"{fsid}-5-99-0.wav"
    frame = pd.concat([frame, val_row], ignore_index=True)

    assert cross_split_fsids(frame, _config()) == [fsid]
    filtered, excluded = filter_cross_split_recordings(frame, _config())
    assert excluded == [fsid]
    assert fsid not in set(filtered["fsID"])
    # We drop the recording rather than moving either occurrence to another fold.
    assert set(filtered["fold"]).issubset(set(range(1, 11)))


def test_manifest_builder_excludes_cross_split_fsids_and_records_policy(tmp_path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "manifests"
    _write_fake_dataset(dataset_root)
    path = dataset_root / "metadata" / "UrbanSound8K.csv"
    frame = pd.read_csv(path)

    train_row = frame[(frame["fold"] == 1) & (frame["class"] == "siren")].iloc[[0]].copy()
    fsid = int(train_row.iloc[0]["fsID"])
    val_row = frame[(frame["fold"] == 8) & (frame["class"] == "engine_idling")].iloc[[0]].copy()
    val_row["fsID"] = fsid
    val_row["slice_file_name"] = f"{fsid}-5-99-0.wav"
    frame = pd.concat([frame, val_row], ignore_index=True)
    frame.to_csv(path, index=False)

    paths = build_urbansound8k_manifests(
        dataset_root, output_dir, _config(), include_mixtures=False
    )
    for key in ("known_train", "known_val", "known_test", "ood_val", "ood_test"):
        manifest = pd.read_csv(paths[key])
        assert not manifest["source_a"].astype(str).str.contains(f"{fsid}-").any()

    import json

    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["cross_split_recording_policy"] == "exclude_from_all_research_manifests"
    assert summary["excluded_cross_split_fsids"] == [fsid]
    assert summary["excluded_cross_split_clips"] == 2
