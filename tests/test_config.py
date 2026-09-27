from pathlib import Path

import pytest
import yaml

from esaudio.config import load_config, stream_hop_samples, validate_config, window_samples


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_valid():
    config = load_config(ROOT / "config" / "default.yaml")
    assert config["project"]["sample_rate"] == 22050
    assert window_samples(config) == 44100
    assert stream_hop_samples(config) == 22050
    assert set(config["project"]["target_classes"]).isdisjoint(config["project"]["heldout_classes"])


def test_demo_config_is_valid():
    config = load_config(ROOT / "config" / "demo.yaml")
    assert len(config["project"]["target_classes"]) == 4
    assert window_samples(config) == 8000
    assert stream_hop_samples(config) == 4000


def test_missing_config_section_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("project: {sample_rate: 8000}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required configuration section"):
        load_config(path)


def test_overlapping_folds_fail():
    config = load_config(ROOT / "config" / "demo.yaml")
    config["data"]["test_folds"] = [2]
    with pytest.raises(ValueError, match="must be disjoint"):
        validate_config(config)


def test_target_and_heldout_classes_must_be_disjoint():
    config = load_config(ROOT / "config" / "demo.yaml")
    config["project"]["heldout_classes"] = [config["project"]["target_classes"][0]]
    with pytest.raises(ValueError, match="must be disjoint"):
        validate_config(config)


def test_feature_fmax_must_not_exceed_nyquist():
    config = load_config(ROOT / "config" / "demo.yaml")
    config["features"]["f_max"] = 5000.0
    with pytest.raises(ValueError, match="Nyquist"):
        validate_config(config)


def test_hop_must_not_exceed_window():
    config = load_config(ROOT / "config" / "demo.yaml")
    config["project"]["hop_seconds"] = 2.0
    with pytest.raises(ValueError, match="must not exceed"):
        validate_config(config)


def test_invalid_yaml_fails_clearly(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("project: [not: valid", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(path)
