"""Regression tests for checkpoint, file-window, and manifest handling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from esaudio.checkpoints import load_checkpoint, save_checkpoint, save_thresholds
from esaudio.config import load_config
from esaudio.dataset import AudioManifestDataset
from esaudio.features import feature_extractor_from_config
from esaudio.inference import AudioTagger
from esaudio.models import build_model
from esaudio.runtime_profile import RuntimeProfiler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkpoint_pair(tmp_path):
    config = load_config(ROOT / "config" / "demo.yaml")
    names = list(config["project"]["target_classes"])
    model, spec = build_model(config, "cnn", len(names))
    extractor = feature_extractor_from_config(config)
    checkpoint = tmp_path / "checkpoint.pt"
    thresholds = tmp_path / "thresholds.json"
    save_checkpoint(
        checkpoint,
        model,
        spec.as_dict(),
        names,
        extractor.export_config(),
        {
            "sample_rate": 8000,
            "window_seconds": 1.0,
            "hop_seconds": 0.5,
            "target_rms_dbfs": -20.0,
        },
        training_metadata={"seed": 13},
    )
    save_thresholds(thresholds, names, [0.5] * len(names))
    return checkpoint, thresholds


def test_checkpoint_loads_using_weights_only(checkpoint_pair, monkeypatch):
    checkpoint, _ = checkpoint_pair
    original_load = torch.load
    calls = []

    def guarded_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded_load)
    model, payload = load_checkpoint(checkpoint)
    assert model.training is False
    assert payload["model_spec"]["name"] == "cnn"
    assert calls == [True]


def test_runtime_profiler_loads_checkpoint_once(checkpoint_pair, monkeypatch):
    checkpoint, thresholds = checkpoint_pair
    original_load = torch.load
    calls = []

    def counting_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)
    profiler = RuntimeProfiler(checkpoint, thresholds, device="cpu")
    assert calls == [True]
    assert profiler.model_spec["name"] == "cnn"
    assert profiler.training_metadata["seed"] == 13
    assert profiler.hop_seconds == pytest.approx(0.5)


def test_file_inference_uses_fixed_hops_and_padded_tail(checkpoint_pair, tmp_path):
    checkpoint, thresholds = checkpoint_pair
    tagger = AudioTagger.from_files(checkpoint, thresholds)
    wave_path = tmp_path / "two_and_half_seconds.wav"
    sf.write(wave_path, np.zeros(20_000, dtype=np.float32), 8_000)

    windows = tagger.predict_file_windows(wave_path)
    assert [w["window_index"] for w in windows] == list(range(5))
    assert [w["start_seconds"] for w in windows] == pytest.approx(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )
    assert [w["end_seconds"] for w in windows] == pytest.approx(
        [1.0, 1.5, 2.0, 2.5, 2.5]
    )


@pytest.mark.parametrize("hop", [0, -0.5, float("nan"), float("inf"), 2.0])
def test_file_inference_rejects_invalid_hops(checkpoint_pair, tmp_path, hop):
    checkpoint, thresholds = checkpoint_pair
    tagger = AudioTagger.from_files(checkpoint, thresholds)
    wave_path = tmp_path / "file.wav"
    sf.write(wave_path, np.zeros(8_000, dtype=np.float32), 8_000)
    with pytest.raises(ValueError, match="hop_seconds"):
        tagger.predict_file_windows(wave_path, hop_seconds=hop)


def _ood_csv(tmp_path: Path, values: list[str]) -> Path:
    frame = pd.DataFrame(
        {
            "sample_id": [f"x_{i}" for i in range(len(values))],
            "split": ["train"] * len(values),
            "is_ood": values,
        }
    )
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    return path


def test_dataset_excludes_only_true_ood_flags(tmp_path):
    config = load_config(ROOT / "config" / "demo.yaml")
    path = _ood_csv(tmp_path, ["False", "True", "0"])
    dataset = AudioManifestDataset(path, tmp_path, config, split="train")
    assert len(dataset) == 2
    assert dataset.frame["sample_id"].tolist() == ["x_0", "x_2"]
    with_ood = AudioManifestDataset(path, tmp_path, config, split="train", include_ood=True)
    assert len(with_ood) == 3


def test_dataset_rejects_ambiguous_ood_flags(tmp_path):
    config = load_config(ROOT / "config" / "demo.yaml")
    path = _ood_csv(tmp_path, ["False", "unknown"])
    with pytest.raises(ValueError, match="Invalid is_ood"):
        AudioManifestDataset(path, tmp_path, config, split="train")
