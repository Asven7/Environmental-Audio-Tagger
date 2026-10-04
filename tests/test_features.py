from __future__ import annotations

import pytest
import torch

from esaudio.features import LogMelExtractor, feature_extractor_from_config


def _default_extractor(n_mels: int = 64, normalize: bool = True) -> LogMelExtractor:
    return LogMelExtractor(
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=n_mels,
        f_min=20.0,
        f_max=10000.0,
        normalize_features=normalize,
    )


def test_two_second_default_geometry_is_fixed() -> None:
    extractor = _default_extractor()
    waveform = torch.randn(2, 44100)
    features = extractor(waveform)
    assert extractor.expected_num_frames(44100) == 87
    assert features.shape == (2, 1, 64, 87)


def test_single_waveform_is_promoted_to_batch() -> None:
    extractor = _default_extractor()
    features = extractor(torch.randn(44100))
    assert features.shape == (1, 1, 64, 87)


def test_silence_is_finite_after_log_and_normalization() -> None:
    extractor = _default_extractor()
    features = extractor(torch.zeros(44100))
    assert torch.isfinite(features).all()
    assert float(features.abs().max()) < 1e-4


def test_output_is_deterministic() -> None:
    torch.manual_seed(123)
    waveform = torch.randn(2, 44100)
    extractor = _default_extractor()
    first = extractor(waveform)
    second = extractor(waveform)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_normalized_features_have_zero_mean_and_unit_std() -> None:
    torch.manual_seed(456)
    features = _default_extractor()(torch.randn(3, 44100)).squeeze(1)
    mean = features.mean(dim=(-2, -1))
    std = features.std(dim=(-2, -1))
    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(std, torch.ones_like(std), atol=1e-5, rtol=0.0)


def test_128_mels_changes_only_frequency_axis() -> None:
    waveform = torch.randn(2, 44100)
    f64 = _default_extractor(64)(waveform)
    f128 = _default_extractor(128)(waveform)
    assert f64.shape == (2, 1, 64, 87)
    assert f128.shape == (2, 1, 128, 87)


def test_export_config_round_trip_preserves_features() -> None:
    torch.manual_seed(789)
    waveform = torch.randn(1, 44100)
    first = _default_extractor()
    second = LogMelExtractor(**first.export_config())
    torch.testing.assert_close(first(waveform), second(waveform), rtol=0.0, atol=0.0)


def test_feature_extractor_from_config_keeps_project_defaults() -> None:
    config = {
        "project": {"sample_rate": 22050},
        "features": {
            "n_fft": 1024,
            "hop_length": 512,
            "n_mels": 64,
            "f_min": 20.0,
            "f_max": 10000.0,
            "normalize_features": True,
        },
    }
    extractor = feature_extractor_from_config(config)
    exported = extractor.export_config()
    assert exported["center"] is True
    assert exported["power"] == 2.0
    assert exported["log_floor"] == 1e-6
    assert exported["mel_scale"] == "htk"
    assert exported["mel_norm"] is None


def test_invalid_waveform_rank_raises() -> None:
    extractor = _default_extractor()
    with pytest.raises(ValueError, match="Expected waveform shape"):
        extractor(torch.randn(2, 1, 44100))


def test_constructor_rejects_invalid_frequency_and_hop_settings() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        LogMelExtractor(22050, 1024, 512, 64, f_max=12000)
    with pytest.raises(ValueError, match="hop_length"):
        LogMelExtractor(22050, 1024, 2048, 64, f_max=10000)


def test_extractor_has_no_trainable_parameters() -> None:
    extractor = _default_extractor()
    assert sum(parameter.numel() for parameter in extractor.parameters()) == 0
