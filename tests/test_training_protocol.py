from __future__ import annotations

import inspect
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from esaudio.config import load_config
from esaudio.dataset import DeterministicAugmenter
from esaudio.training import (
    EarlyStoppingState,
    compute_pos_weight,
    make_dataloader_generator,
    seed_everything,
    train_model,
)


ROOT = Path(__file__).resolve().parents[1]


def _shuffled_order(seed: int) -> list[int]:
    dataset = TensorDataset(torch.arange(20))
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        generator=make_dataloader_generator(seed),
    )
    return [int(value) for batch in loader for value in batch[0]]


def test_seed_everything_repeats_python_numpy_and_torch():
    seed_everything(123)
    first = (
        random.random(),
        float(np.random.random()),
        torch.rand(5),
    )
    seed_everything(123)
    second = (
        random.random(),
        float(np.random.random()),
        torch.rand(5),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_dataloader_shuffle_is_reproducible_for_same_seed():
    assert _shuffled_order(13) == _shuffled_order(13)


def test_dataloader_shuffle_changes_for_different_seed():
    assert _shuffled_order(13) != _shuffled_order(23)


def test_dataloader_shuffle_is_independent_of_global_rng_consumption():
    seed_everything(999)
    _ = torch.rand(1000)
    first = _shuffled_order(13)

    seed_everything(7)
    _ = torch.rand(3)
    second = _shuffled_order(13)

    assert first == second


def test_compute_pos_weight_uses_standard_negative_over_positive_formula():
    targets = np.array(
        [
            [1, 0],
            [1, 1],
            [0, 0],
            [0, 0],
        ],
        dtype=np.float32,
    )
    weight = compute_pos_weight(targets)
    assert torch.allclose(weight, torch.tensor([1.0, 3.0]))


def test_compute_pos_weight_rejects_class_missing_from_training():
    targets = np.array([[1, 0], [0, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="no positive example"):
        compute_pos_weight(targets)


def test_early_stopping_tracks_best_validation_metric_and_patience():
    state = EarlyStoppingState(patience=2)
    assert state.update(0.20, 1) is True
    assert state.best_epoch == 1
    assert state.should_stop is False

    assert state.update(0.19, 2) is False
    assert state.should_stop is False

    assert state.update(0.20, 3) is False
    assert state.should_stop is True
    assert state.best_epoch == 1
    assert state.best_value == pytest.approx(0.20)


def test_train_augmentation_is_deterministic_per_sample_and_epoch():
    config = load_config(ROOT / "config" / "default.yaml")
    augmenter = DeterministicAugmenter(config)
    waveform = np.linspace(-0.4, 0.4, 44100, dtype=np.float32)

    first = augmenter(waveform, "sample-a", epoch=1)
    second = augmenter(waveform, "sample-a", epoch=1)
    later_epoch = augmenter(waveform, "sample-a", epoch=2)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, later_epoch)


def test_training_defaults_keep_backward_compatible_threshold_behavior():
    signature = inspect.signature(train_model)
    assert signature.parameters["tune_thresholds"].default is True


def test_research_experiment_runner_never_reads_test_or_ood_manifests():
    source = (ROOT / "scripts" / "run_experiments.py").read_text(encoding="utf-8")

    forbidden = (
        "known_test.csv",
        "ood_test.csv",
        "evaluate_checkpoint",
        "test_evaluation.json",
    )
    for token in forbidden:
        assert token not in source

    assert "known_train.csv" in source
    assert "known_val.csv" in source
    assert "tune_thresholds=False" in source


def test_default_training_contract_values():
    config = load_config(ROOT / "config" / "default.yaml")
    training = config["training"]

    assert training["batch_size"] == 32
    assert training["epochs"] == 30
    assert training["learning_rate"] == pytest.approx(0.001)
    assert training["weight_decay"] == pytest.approx(0.0001)
    assert training["early_stopping_patience"] == 5
    assert training["use_pos_weight"] is True
    assert training["data_seed"] == 1234
    assert training["seeds"] == [13, 23, 37]
