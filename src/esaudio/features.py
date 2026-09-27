from __future__ import annotations

import torch
from torch import nn
import torchaudio


class LogMelExtractor(nn.Module):
    """Log-Mel feature extractor shared by training and inference."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float = 20.0,
        f_max: float | None = None,
        normalize_features: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = float(f_max) if f_max is not None else self.sample_rate / 2
        self.normalize_features = bool(normalize_features)

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            power=2.0,
            center=True,
            norm=None,
            mel_scale="htk",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert [T] or [B,T] waveforms to [B,1,M,Tf] log-Mel features."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform shape [T] or [B,T], got {tuple(waveform.shape)}")

        mel = self.mel(waveform)
        log_mel = torch.log(mel.clamp_min(1e-6))
        if self.normalize_features:
            mean = log_mel.mean(dim=(-2, -1), keepdim=True)
            std = log_mel.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
            log_mel = (log_mel - mean) / std
        return log_mel.unsqueeze(1)

    def export_config(self) -> dict[str, int | float | bool]:
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "normalize_features": self.normalize_features,
        }


def feature_extractor_from_config(config: dict) -> LogMelExtractor:
    project = config["project"]
    features = config["features"]
    return LogMelExtractor(
        sample_rate=int(project["sample_rate"]),
        n_fft=int(features["n_fft"]),
        hop_length=int(features["hop_length"]),
        n_mels=int(features["n_mels"]),
        f_min=float(features.get("f_min", 20.0)),
        f_max=float(features.get("f_max", int(project["sample_rate"]) / 2)),
        normalize_features=bool(features.get("normalize_features", True)),
    )
