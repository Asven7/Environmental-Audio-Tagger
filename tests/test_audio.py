import numpy as np

from esaudio.audio import mix_two_sources, normalize_rms, pad_or_crop, rms_dbfs


def test_pad_or_crop_lengths():
    short = np.ones(10, dtype=np.float32)
    assert pad_or_crop(short, 20).shape == (20,)
    long = np.arange(100, dtype=np.float32)
    assert pad_or_crop(long, 20, "center").shape == (20,)
    assert pad_or_crop(long, 20, "energy").shape == (20,)


def test_normalize_rms_moves_toward_target():
    x = np.ones(1000, dtype=np.float32) * 0.05
    y = normalize_rms(x, target_dbfs=-20.0, max_gain_db=20.0)
    assert abs(rms_dbfs(y) - (-20.0)) < 0.2


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
