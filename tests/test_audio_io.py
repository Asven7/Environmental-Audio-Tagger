from pathlib import Path

import numpy as np
import soundfile as sf

from esaudio.audio import load_audio


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
    assert np.isfinite(waveform).all()
