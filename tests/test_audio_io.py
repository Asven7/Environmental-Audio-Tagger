import numpy as np
import pytest
import soundfile as sf

from esaudio.audio import load_audio, resample_audio, save_audio


def test_load_audio_downmixes_and_resamples(tmp_path):
    source_sr = 16000
    target_sr = 8000
    t = np.arange(source_sr, dtype=np.float32) / source_sr
    left = 0.1 * np.sin(2 * np.pi * 220 * t)
    right = 0.1 * np.sin(2 * np.pi * 330 * t)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, source_sr)

    waveform = load_audio(path, target_sr)
    assert waveform.ndim == 1
    assert abs(len(waveform) - target_sr) <= 2
    assert waveform.dtype == np.float32
    assert np.isfinite(waveform).all()


def test_resample_same_rate_returns_copy():
    x = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
    y = resample_audio(x, 8000, 8000)
    np.testing.assert_array_equal(x, y)
    assert y is not x


def test_missing_audio_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        load_audio(tmp_path / "missing.wav", 8000)


def test_save_and_load_roundtrip(tmp_path):
    sample_rate = 8000
    waveform = np.linspace(-0.25, 0.25, sample_rate, dtype=np.float32)
    path = tmp_path / "roundtrip.wav"
    save_audio(path, waveform, sample_rate)
    loaded = load_audio(path, sample_rate)
    assert loaded.shape == waveform.shape
    assert np.max(np.abs(loaded - waveform)) < 1e-4
