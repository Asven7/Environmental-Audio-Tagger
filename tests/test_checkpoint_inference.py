from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from esaudio.checkpoints import save_checkpoint, save_thresholds
from esaudio.config import load_config
from esaudio.features import feature_extractor_from_config
from esaudio.inference import AudioTagger
from esaudio.models import build_model


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_roundtrip_and_inference(tmp_path):
    config = load_config(ROOT / "config" / "demo.yaml")
    model, spec = build_model(config, "crnn", 4)
    extractor = feature_extractor_from_config(config)
    checkpoint = tmp_path / "model.pt"
    thresholds = tmp_path / "thresholds.json"
    save_checkpoint(
        checkpoint,
        model,
        spec.as_dict(),
        config["project"]["target_classes"],
        extractor.export_config(),
        {
            "sample_rate": 8000,
            "window_seconds": 1.0,
            "hop_seconds": 0.5,
            "target_rms_dbfs": -20.0,
        },
    )
    save_thresholds(thresholds, config["project"]["target_classes"], [0.5] * 4)
    tagger = AudioTagger.from_files(checkpoint, thresholds)
    result = tagger.predict_waveform(np.zeros(8000, dtype=np.float32))
    assert set(result.scores) == set(config["project"]["target_classes"])
    assert result.total_ms >= 0

    audio_path = tmp_path / "long.wav"
    sf.write(audio_path, np.zeros(16000, dtype=np.float32), 8000)
    windows = tagger.predict_file_windows(audio_path)
    assert len(windows) == 3  # 1 s window, 0.5 s hop over a 2 s file
    assert windows[0]["start_seconds"] == 0.0
    assert windows[-1]["end_seconds"] == 2.0
