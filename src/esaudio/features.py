from __future__ import annotations

import torch
from torch import nn
import torchaudio


class LogMelExtractor(nn.Module):
    """Log-Mel feature extractor shared by training and inference.

    The public constructor remains backward compatible with earlier checkpoints:
    newly exposed reproducibility knobs have defaults matching the original
    implementation used by this project.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float = 20.0,
        f_max: float | None = None,
        normalize_features: bool = True,
        center: bool = True,
        power: float = 2.0,
        log_floor: float = 1e-6,
        mel_scale: str = "htk",
        mel_norm: str | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = float(f_max) if f_max is not None else self.sample_rate / 2
        self.normalize_features = bool(normalize_features)
        self.center = bool(center)
        self.power = float(power)
        self.log_floor = float(log_floor)
        self.mel_scale = str(mel_scale)
        self.mel_norm = mel_norm

        self._validate_parameters()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            power=self.power,
            center=self.center,
            norm=self.mel_norm,
            mel_scale=self.mel_scale,
        )

    def _validate_parameters(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.hop_length <= 0 or self.hop_length > self.n_fft:
            raise ValueError("hop_length must be positive and <= n_fft")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if self.f_min < 0.0:
            raise ValueError("f_min must be >= 0")
        if self.f_max <= self.f_min:
            raise ValueError("f_max must be greater than f_min")
        if self.f_max > self.sample_rate / 2:
            raise ValueError("f_max must not exceed the Nyquist frequency")
        if self.power <= 0.0:
            raise ValueError("power must be positive")
        if self.log_floor <= 0.0:
            raise ValueError("log_floor must be positive")
        if self.mel_scale not in {"htk", "slaney"}:
            raise ValueError("mel_scale must be 'htk' or 'slaney'")
        if self.mel_norm not in {None, "slaney"}:
            raise ValueError("mel_norm must be None or 'slaney'")

    def expected_num_frames(self, num_samples: int) -> int:
        """Return the deterministic STFT frame count for a fixed input length."""
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        pad = self.n_fft // 2 if self.center else 0
        effective = num_samples + 2 * pad
        if effective < self.n_fft:
            return 0
        return 1 + (effective - self.n_fft) // self.hop_length

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert [T] or [B,T] waveforms to [B,1,M,Tf] Log-Mel features."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(
                f"Expected waveform shape [T] or [B,T], got {tuple(waveform.shape)}"
            )
        if waveform.shape[-1] == 0:
            raise ValueError("waveform must contain at least one sample")
        if not torch.is_floating_point(waveform):
            waveform = waveform.float()

        mel = self.mel(waveform)
        log_mel = torch.log(mel.clamp_min(self.log_floor))
        if self.normalize_features:
            mean = log_mel.mean(dim=(-2, -1), keepdim=True)
            raw_std = log_mel.std(dim=(-2, -1), keepdim=True)
            normalized = (log_mel - mean) / raw_std.clamp_min(1e-5)
            log_mel = torch.where(raw_std < 1e-5, torch.zeros_like(normalized), normalized)
        return log_mel.unsqueeze(1)

    def export_config(self) -> dict[str, int | float | bool | str | None]:
        """Export every transform choice needed to reconstruct identical features."""
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "normalize_features": self.normalize_features,
            "center": self.center,
            "power": self.power,
            "log_floor": self.log_floor,
            "mel_scale": self.mel_scale,
            "mel_norm": self.mel_norm,
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
        center=bool(features.get("center", True)),
        power=float(features.get("power", 2.0)),
        log_floor=float(features.get("log_floor", 1e-6)),
        mel_scale=str(features.get("mel_scale", "htk")),
        mel_norm=features.get("mel_norm", None),
    )
