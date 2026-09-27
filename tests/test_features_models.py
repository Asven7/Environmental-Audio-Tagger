from pathlib import Path

import torch

from esaudio.config import load_config
from esaudio.features import feature_extractor_from_config
from esaudio.models import build_model, count_parameters


ROOT = Path(__file__).resolve().parents[1]


def test_features_and_models_forward():
    config = load_config(ROOT / "config" / "demo.yaml")
    extractor = feature_extractor_from_config(config)
    waveform = torch.randn(2, 8000)
    features = extractor(waveform)
    assert features.shape[0] == 2
    assert features.shape[1] == 1
    assert features.shape[2] == 32

    for model_name in ("cnn", "crnn"):
        model, _ = build_model(config, model_name, 4)
        logits = model(features)
        assert logits.shape == (2, 4)
        assert count_parameters(model) > 0
