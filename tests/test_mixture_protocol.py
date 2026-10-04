import json

import numpy as np
import pandas as pd
import pytest

from esaudio.audio import mix_two_sources
from esaudio.manifests import (
    MANIFEST_COLUMNS,
    generate_mixture_rows,
    parse_json_list,
    validate_mixture_rows,
    write_manifests,
)


TARGET_CLASSES = [
    "air_conditioner",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "jackhammer",
    "siren",
    "car_horn",
]
DB_LEVELS = [-6.0, 0.0, 6.0]
OVERLAPS = [0.25, 0.5, 1.0]


def _known_singles(groups_per_class: int = 12, splits=("train",)) -> pd.DataFrame:
    rows = []
    sample_no = 0
    for split_no, split in enumerate(splits):
        for class_index, class_name in enumerate(TARGET_CLASSES):
            for group_no in range(groups_per_class):
                sample_no += 1
                fsid = 100000 + split_no * 10000 + class_index * 100 + group_no
                rows.append(
                    {
                        "sample_id": f"single-{split}-{sample_no:05d}",
                        "split": split,
                        "sample_type": "single",
                        "source_a": f"audio/{split}/{fsid}-{class_index}-0-0.wav",
                        "source_b": "",
                        "source_group_a": f"occurrence:{fsid}:{class_index}:0",
                        "source_group_b": "",
                        "label_indices": json.dumps([class_index]),
                        "label_names": json.dumps([class_name]),
                        "relative_db": np.nan,
                        "overlap_ratio": 0.0,
                        "is_ood": False,
                    }
                )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _config(train_count=56, val_count=28, test_count=28) -> dict:
    return {
        "project": {
            "target_classes": TARGET_CLASSES,
            "heldout_classes": ["gun_shot", "street_music"],
        },
        "mixing": {
            "relative_db_levels": DB_LEVELS,
            "overlap_ratios": OVERLAPS,
            "train_mixtures": train_count,
            "val_mixtures": val_count,
            "test_mixtures": test_count,
        },
        "training": {"data_seed": 1234},
    }


def test_mixture_generation_is_deterministic_and_exact_count():
    singles = _known_singles()
    kwargs = dict(
        singles=singles,
        target_classes=TARGET_CLASSES,
        split="train",
        count=280,
        relative_db_levels=DB_LEVELS,
        overlap_ratios=OVERLAPS,
        seed=1234,
    )
    first = generate_mixture_rows(**kwargs)
    second = generate_mixture_rows(**kwargs)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 280
    assert first["sample_id"].is_unique


def test_mixture_protocol_balances_class_pairs_conditions_and_orientation():
    singles = _known_singles()
    mixtures = generate_mixture_rows(
        singles, TARGET_CLASSES, "train", 280, DB_LEVELS, OVERLAPS, 1234
    )
    audit = validate_mixture_rows(
        mixtures,
        TARGET_CLASSES,
        DB_LEVELS,
        OVERLAPS,
        expected_split="train",
        source_singles=singles,
    )
    assert audit["class_pair_count_min"] == 10
    assert audit["class_pair_count_max"] == 10
    assert audit["condition_count_max"] - audit["condition_count_min"] <= 1
    assert audit["unique_source_group_pairs"] == 280


def test_mixture_targets_equal_source_label_union_and_avoid_same_fsid():
    singles = _known_singles()
    mixtures = generate_mixture_rows(
        singles, TARGET_CLASSES, "train", 140, DB_LEVELS, OVERLAPS, 77
    )
    source_to_label = {
        row.source_a: parse_json_list(row.label_names)[0]
        for row in singles.itertuples(index=False)
    }
    seen = set()
    for row in mixtures.itertuples(index=False):
        labels = parse_json_list(row.label_names)
        assert len(labels) == 2
        assert len(set(labels)) == 2
        assert set(labels) == {source_to_label[row.source_a], source_to_label[row.source_b]}
        fsid_a = int(str(row.source_group_a).split(":")[1])
        fsid_b = int(str(row.source_group_b).split(":")[1])
        assert fsid_a != fsid_b
        pair = tuple(sorted((row.source_group_a, row.source_group_b)))
        assert pair not in seen
        seen.add(pair)


def test_mixture_validator_rejects_duplicate_source_group_pair():
    singles = _known_singles()
    mixtures = generate_mixture_rows(
        singles, TARGET_CLASSES, "train", 28, DB_LEVELS, OVERLAPS, 9
    )
    duplicate = mixtures.iloc[[0]].copy()
    duplicate["sample_id"] = "mix-train-duplicate"
    broken = pd.concat([mixtures, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate source-group pair"):
        validate_mixture_rows(
            broken,
            TARGET_CLASSES,
            DB_LEVELS,
            OVERLAPS,
            expected_split="train",
            source_singles=singles,
            enforce_balance=False,
        )


def test_write_manifests_records_reproducible_mixture_protocol(tmp_path):
    singles = _known_singles(groups_per_class=8, splits=("train", "val", "test"))
    ood = pd.DataFrame(columns=MANIFEST_COLUMNS)
    paths = write_manifests(singles, ood, tmp_path, _config(), include_mixtures=True)

    train = pd.read_csv(paths["known_train"])
    val = pd.read_csv(paths["known_val"])
    test = pd.read_csv(paths["known_test"])
    assert (train["sample_type"] == "mix").sum() == 56
    assert (val["sample_type"] == "mix").sum() == 28
    assert (test["sample_type"] == "mix").sum() == 28

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    protocol = summary["mixture_protocol"]
    assert protocol["generator"] == "balanced_pair_condition_v1"
    assert protocol["data_seed"] == 1234
    assert protocol["duplicate_source_group_pairs_allowed"] is False
    assert protocol["same_broad_fsid_pairing_allowed_when_identifiable"] is False
    assert protocol["split_audits"]["train"]["rows"] == 56



def test_tiny_path_fallback_demo_can_reuse_source_pairs_without_weakening_research_rule(tmp_path):
    """Demo smoke data may have fewer unique pairs than its configured mixture count."""
    demo_classes = ["demo_a", "demo_b", "demo_c", "demo_d"]
    rows = []
    for class_index, class_name in enumerate(demo_classes):
        for item in range(2):
            rows.append(
                {
                    "sample_id": f"demo-train-{class_name}-{item}",
                    "split": "train",
                    "sample_type": "single",
                    "source_a": f"audio/train/{class_name}/{item:03d}.wav",
                    "source_b": "",
                    "source_group_a": "",
                    "source_group_b": "",
                    "label_indices": json.dumps([class_index]),
                    "label_names": json.dumps([class_name]),
                    "relative_db": np.nan,
                    "overlap_ratio": 0.0,
                    "is_ood": False,
                }
            )
    singles = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    # There are only 24 distinct cross-class file pairs (6 class pairs × 2 × 2),
    # so a request for 36 mixtures necessarily requires pair reuse.
    config = {
        "project": {"target_classes": demo_classes, "heldout_classes": []},
        "mixing": {
            "relative_db_levels": [-6.0, 0.0, 6.0],
            "overlap_ratios": [0.5, 1.0],
            "train_mixtures": 36,
            "val_mixtures": 0,
            "test_mixtures": 0,
        },
        "training": {"data_seed": 1234},
    }
    paths = write_manifests(
        singles, pd.DataFrame(columns=MANIFEST_COLUMNS), tmp_path, config, include_mixtures=True
    )
    train = pd.read_csv(paths["known_train"])
    mixtures = train[train["sample_type"] == "mix"].reset_index(drop=True)
    audit = validate_mixture_rows(
        mixtures,
        demo_classes,
        [-6.0, 0.0, 6.0],
        [0.5, 1.0],
        expected_split="train",
        source_singles=singles,
    )
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert len(mixtures) == 36
    assert audit["unique_source_pairs_required"] is False
    assert audit["unique_source_group_pairs"] < len(mixtures)
    assert summary["mixture_protocol"]["duplicate_source_group_pairs_allowed"] is True
    assert summary["mixture_protocol"]["research_occurrence_pairs_remain_unique"] is True


@pytest.mark.parametrize("relative_db", [-6.0, 0.0, 6.0])
def test_waveform_mixing_realizes_requested_relative_db(relative_db):
    sample_rate = 8000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    first = (0.02 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    second = (0.2 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    mixed = mix_two_sources(first, second, relative_db=relative_db, overlap_ratio=1.0)

    spectrum = np.abs(np.fft.rfft(mixed))
    freqs = np.fft.rfftfreq(len(mixed), 1 / sample_rate)
    amp_first = spectrum[np.argmin(np.abs(freqs - 250))]
    amp_second = spectrum[np.argmin(np.abs(freqs - 500))]
    measured_db = 20 * np.log10(amp_first / amp_second)
    assert abs(measured_db - relative_db) < 0.5
    assert np.max(np.abs(mixed)) <= 0.991
