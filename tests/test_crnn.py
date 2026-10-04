from pathlib import Path

import pytest
import torch
from torch import nn

from esaudio.config import load_config
from esaudio.models import CRNNTagger, build_model, count_parameters, model_from_spec


ROOT = Path(__file__).resolve().parents[1]


def _default_crnn() -> tuple[CRNNTagger, dict]:
    config = load_config(ROOT / "config" / "default.yaml")
    model, _ = build_model(config, "crnn", len(config["project"]["target_classes"]))
    assert isinstance(model, CRNNTagger)
    return model, config


def test_default_crnn_output_shape_and_finite_logits():
    model, config = _default_crnn()
    features = torch.randn(3, 1, int(config["features"]["n_mels"]), 87)
    logits = model(features)
    assert logits.shape == (3, len(config["project"]["target_classes"]))
    assert torch.isfinite(logits).all()


def test_crnn_sequence_contract_preserves_87_time_steps():
    model, _ = _default_crnn()
    features = torch.randn(2, 1, 64, 87)
    sequence = model.sequence_features(features)
    assert sequence.shape == (2, 87, 64)

    recurrent, _ = model.rnn(sequence)
    assert recurrent.shape == (2, 87, 64)


def test_default_crnn_is_single_layer_unidirectional_gru():
    model, _ = _default_crnn()
    assert isinstance(model.rnn, nn.GRU)
    assert model.recurrent_type == "gru"
    assert model.recurrent_hidden == 64
    assert model.recurrent_layers == 1
    assert model.rnn.bidirectional is False
    assert model.rnn.num_layers == 1
    assert model.rnn.dropout == 0.0


def test_crnn_supports_batch_size_one():
    model, _ = _default_crnn()
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 1, 64, 87))
    assert logits.shape == (1, 8)


def test_crnn_supports_128_mel_bins_without_architecture_change():
    model, _ = _default_crnn()
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 1, 128, 87))
    assert logits.shape == (2, 8)


def test_crnn_returns_raw_logits_and_contains_no_sigmoid():
    model, _ = _default_crnn()
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


def test_crnn_bce_with_logits_backward_has_finite_gradients():
    torch.manual_seed(123)
    model, _ = _default_crnn()
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


def test_default_gru_crnn_parameter_count_is_frozen():
    model, _ = _default_crnn()
    assert count_parameters(model) == 48888


def test_lstm_variant_is_supported_and_parameter_count_is_known():
    model = CRNNTagger(
        num_classes=8,
        channels=[16, 32, 64],
        recurrent_type="lstm",
        recurrent_hidden=64,
        recurrent_layers=1,
        dropout=0.2,
    )
    assert isinstance(model.rnn, nn.LSTM)
    assert count_parameters(model) == 57208
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 1, 64, 87))
    assert logits.shape == (2, 8)


def test_crnn_eval_forward_is_deterministic():
    torch.manual_seed(7)
    model, _ = _default_crnn()
    model.eval()
    features = torch.randn(2, 1, 64, 87)

    with torch.no_grad():
        first = model(features)
        second = model(features)

    assert torch.equal(first, second)


def test_crnn_model_spec_roundtrip_preserves_outputs():
    torch.manual_seed(11)
    config = load_config(ROOT / "config" / "default.yaml")
    model, spec = build_model(config, "crnn", 8)
    restored = model_from_spec(spec.as_dict())
    restored.load_state_dict(model.state_dict())

    model.eval()
    restored.eval()
    features = torch.randn(2, 1, 64, 87)

    with torch.no_grad():
        expected = model(features)
        actual = restored(features)

    assert torch.equal(expected, actual)
    assert count_parameters(restored) == 48888


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"recurrent_type": "rnn"}, "recurrent_type"),
        ({"recurrent_hidden": 0}, "recurrent_hidden"),
        ({"recurrent_layers": 0}, "recurrent_layers"),
    ],
)
def test_invalid_recurrent_configuration_is_rejected(kwargs, message):
    base = {
        "num_classes": 8,
        "channels": [16, 32, 64],
        "recurrent_type": "gru",
        "recurrent_hidden": 64,
        "recurrent_layers": 1,
        "dropout": 0.2,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        CRNNTagger(**base)
