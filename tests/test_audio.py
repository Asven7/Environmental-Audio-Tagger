import numpy as np
import pytest

from esaudio.audio import apply_global_time_shift, mix_two_sources, normalize_rms, pad_or_crop, rms_dbfs


def test_pad_or_crop_lengths_and_right_padding():
    short = np.arange(10, dtype=np.float32)
    padded = pad_or_crop(short, 20)
    assert padded.shape == (20,)
    np.testing.assert_array_equal(padded[:10], short)
    np.testing.assert_array_equal(padded[10:], np.zeros(10, dtype=np.float32))

    long = np.arange(100, dtype=np.float32)
    assert pad_or_crop(long, 20, "center").shape == (20,)
    assert pad_or_crop(long, 20, "energy").shape == (20,)


def test_center_crop_is_deterministic():
    waveform = np.arange(10, dtype=np.float32)
    cropped = pad_or_crop(waveform, 4, strategy="center")
    np.testing.assert_array_equal(cropped, np.array([3, 4, 5, 6], dtype=np.float32))


def test_energy_crop_selects_high_energy_region():
    waveform = np.zeros(20, dtype=np.float32)
    waveform[12:16] = 1.0
    cropped = pad_or_crop(waveform, 4, strategy="energy")
    np.testing.assert_array_equal(cropped, np.ones(4, dtype=np.float32))


def test_invalid_crop_strategy_fails():
    with pytest.raises(ValueError, match="strategy"):
        pad_or_crop(np.ones(20, dtype=np.float32), 10, strategy="random")


def test_normalize_rms_moves_toward_target():
    x = np.ones(1000, dtype=np.float32) * 0.05
    y = normalize_rms(x, target_dbfs=-20.0, max_gain_db=20.0)
    assert abs(rms_dbfs(y) - (-20.0)) < 0.2


def test_silence_is_not_amplified():
    x = np.zeros(1000, dtype=np.float32)
    y = normalize_rms(x, target_dbfs=-20.0)
    np.testing.assert_array_equal(y, x)
    assert rms_dbfs(y) == -120.0


def test_mix_preserves_shape_and_no_clipping():
    t = np.linspace(0, 1, 8000, endpoint=False)
    a = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    b = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    mixed = mix_two_sources(a, b, relative_db=6.0, overlap_ratio=0.5)
    assert mixed.shape == a.shape
    assert np.max(np.abs(mixed)) <= 0.991
    assert np.any(np.abs(mixed[4000:]) > 0)


def test_mix_relative_level_is_controlled_for_full_overlap():
    sample_rate = 8000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    a = (0.01 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    b = (0.2 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    mixed = mix_two_sources(a, b, relative_db=6.0, overlap_ratio=1.0)
    spectrum = np.abs(np.fft.rfft(mixed))
    freqs = np.fft.rfftfreq(len(mixed), 1 / sample_rate)
    amp_a = spectrum[np.argmin(np.abs(freqs - 250))]
    amp_b = spectrum[np.argmin(np.abs(freqs - 500))]
    measured_db = 20 * np.log10(amp_a / amp_b)
    assert abs(measured_db - 6.0) < 0.5


def test_mix_rejects_invalid_overlap():
    x = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError, match="overlap_ratio"):
        mix_two_sources(x, x, overlap_ratio=0.0)


def test_global_time_shift_delays_without_wraparound():
    waveform = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    shifted = apply_global_time_shift(waveform, 2)
    np.testing.assert_array_equal(shifted, np.array([0, 0, 1, 2, 3], dtype=np.float32))


def test_global_time_shift_advances_without_wraparound():
    waveform = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    shifted = apply_global_time_shift(waveform, -2)
    np.testing.assert_array_equal(shifted, np.array([3, 4, 5, 0, 0], dtype=np.float32))


def test_global_time_shift_zero_returns_copy_and_large_shift_returns_silence():
    waveform = np.array([1, 2, 3], dtype=np.float32)
    unchanged = apply_global_time_shift(waveform, 0)
    np.testing.assert_array_equal(unchanged, waveform)
    assert unchanged is not waveform

    shifted_out = apply_global_time_shift(waveform, 3)
    np.testing.assert_array_equal(shifted_out, np.zeros_like(waveform))


def test_global_time_shift_rejects_non_integer_shift():
    with pytest.raises(TypeError, match="integer"):
        apply_global_time_shift(np.ones(4, dtype=np.float32), 1.5)
