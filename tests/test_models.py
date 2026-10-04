from pathlib import Path

import pytest
import torch
from torch import nn

from esaudio.config import load_config
from esaudio.models import CNNTagger, build_model, count_parameters, model_from_spec


ROOT = Path(__file__).resolve().parents[1]


def _default_cnn() -> tuple[CNNTagger, dict]:
    config = load_config(ROOT / "config" / "default.yaml")
    model, _ = build_model(config, "cnn", len(config["project"]["target_classes"]))
    assert isinstance(model, CNNTagger)
    return model, config


def test_default_cnn_output_shape():
    model, config = _default_cnn()
    batch = torch.randn(3, 1, int(config["features"]["n_mels"]), 87)
    logits = model(batch)
    assert logits.shape == (3, len(config["project"]["target_classes"]))
    assert torch.isfinite(logits).all()


def test_frontend_preserves_time_axis_and_reduces_only_frequency():
    model, _ = _default_cnn()
    features = torch.randn(2, 1, 64, 87)
    encoded = model.frontend(features)
    assert encoded.shape == (2, 64, 8, 87)


def test_cnn_supports_batch_size_one():
    model, config = _default_cnn()
    model.eval()
    features = torch.randn(1, 1, int(config["features"]["n_mels"]), 87)
    with torch.no_grad():
        logits = model(features)
    assert logits.shape == (1, 8)


def test_cnn_supports_128_mel_bins_without_architecture_change():
    model, _ = _default_cnn()
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 1, 128, 87))
    assert logits.shape == (2, 8)


def test_cnn_returns_raw_logits_and_contains_no_sigmoid():
    model, _ = _default_cnn()
    assert not any(isinstance(module, nn.Sigmoid) for module in model.modules())

    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias.copy_(torch.linspace(-2.0, 2.0, steps=8))
        logits = model(torch.zeros(1, 1, 64, 87))

    assert torch.allclose(logits[0], torch.linspace(-2.0, 2.0, steps=8))
    assert float(logits.min()) < 0.0
    assert float(logits.max()) > 1.0


def test_bce_with_logits_backward_has_finite_gradients():
    torch.manual_seed(123)
    model, _ = _default_cnn()
    model.train()

    features = torch.randn(4, 1, 64, 87)
    targets = torch.randint(0, 2, (4, 8), dtype=torch.float32)
    logits = model(features)
    loss = nn.BCEWithLogitsLoss()(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients
    assert all(grad is not None for grad in gradients)
    assert all(torch.isfinite(grad).all() for grad in gradients if grad is not None)


def test_default_cnn_parameter_count_is_frozen():
    model, _ = _default_cnn()
    assert count_parameters(model) == 23928


def test_eval_forward_is_deterministic():
    torch.manual_seed(7)
    model, _ = _default_cnn()
    model.eval()
    features = torch.randn(2, 1, 64, 87)

    with torch.no_grad():
        first = model(features)
        second = model(features)

    assert torch.equal(first, second)


def test_invalid_feature_rank_is_rejected():
    model, _ = _default_cnn()
    with pytest.raises(ValueError, match=r"\[B,1,M,T\]"):
        model(torch.randn(2, 64, 87))


def test_invalid_feature_channel_count_is_rejected():
    model, _ = _default_cnn()
    with pytest.raises(ValueError, match="one Log-Mel input channel"):
        model(torch.randn(2, 2, 64, 87))


def test_cnn_model_spec_roundtrip_preserves_outputs():
    torch.manual_seed(11)
    config = load_config(ROOT / "config" / "default.yaml")
    model, spec = build_model(config, "cnn", 8)
    restored = model_from_spec(spec.as_dict())
    restored.load_state_dict(model.state_dict())

    model.eval()
    restored.eval()
    features = torch.randn(2, 1, 64, 87)
    with torch.no_grad():
        expected = model(features)
        actual = restored(features)

    assert torch.equal(expected, actual)
    assert count_parameters(restored) == 23928
