from pathlib import Path

import pytest

from esaudio.config import load_config, stream_hop_samples, window_samples


ROOT = Path(__file__).resolve().parents[1]


def test_demo_config_is_valid():
    config = load_config(ROOT / "config" / "demo.yaml")
    assert len(config["project"]["target_classes"]) == 4
    assert window_samples(config) == 8000
    assert stream_hop_samples(config) == 4000


def test_missing_config_key_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("project: {sample_rate: 8000}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
