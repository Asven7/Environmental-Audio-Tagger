from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .audio import load_audio, normalize_rms, pad_or_crop, rms_dbfs
from .checkpoints import load_checkpoint, load_thresholds
from .features import LogMelExtractor
from .streaming import iter_waveform_windows


@dataclass
class Prediction:
    active_labels: list[str]
    scores: dict[str, float]
    status: str
    input_rms_dbfs: float
    feature_ms: float
    inference_ms: float
    total_ms: float

    def as_dict(self) -> dict:
        return {
            "active_labels": self.active_labels,
            "scores": self.scores,
            "status": self.status,
            "input_rms_dbfs": self.input_rms_dbfs,
            "feature_ms": self.feature_ms,
            "inference_ms": self.inference_ms,
            "total_ms": self.total_ms,
        }


class AudioTagger:
    def __init__(
        self,
        model: torch.nn.Module,
        extractor: LogMelExtractor,
        class_names: list[str],
        thresholds: np.ndarray,
        project_config: dict,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.extractor = extractor.to(self.device).eval()
        self.class_names = list(class_names)
        self.thresholds = np.asarray(thresholds, dtype=np.float32)
        if self.thresholds.shape != (len(self.class_names),):
            raise ValueError("Threshold count must match class count")
        self.sample_rate = int(project_config["sample_rate"])
        self.window_seconds = float(project_config["window_seconds"])
        self.hop_seconds = float(project_config.get("hop_seconds", self.window_seconds / 2.0))
        self.window_samples = round(self.sample_rate * self.window_seconds)
        self.target_rms_dbfs = float(project_config.get("target_rms_dbfs", -20.0))
        self.model_spec: dict = {}
        self.training_metadata: dict = {}

    @classmethod
    def from_files(
        cls,
        checkpoint_path: str | Path,
        thresholds_path: str | Path,
        device: str = "cpu",
    ) -> "AudioTagger":
        model, payload = load_checkpoint(checkpoint_path, device=device)
        feature_cfg = payload["feature_config"]
        extractor = LogMelExtractor(**feature_cfg)
        class_names = list(payload["class_names"])
        thresholds = load_thresholds(thresholds_path, expected_classes=class_names)
        tagger = cls(
            model=model,
            extractor=extractor,
            class_names=class_names,
            thresholds=np.asarray(thresholds, dtype=np.float32),
            project_config=payload["project_config"],
            device=device,
        )
        tagger.model_spec = dict(payload["model_spec"])
        tagger.training_metadata = dict(payload.get("training_metadata", {}))
        return tagger

    @torch.no_grad()
    def predict_waveform(self, waveform: np.ndarray) -> Prediction:
        start_total = time.perf_counter()
        waveform = pad_or_crop(np.asarray(waveform, dtype=np.float32), self.window_samples, strategy="center")
        original_rms = rms_dbfs(waveform)
        waveform = normalize_rms(waveform, target_dbfs=self.target_rms_dbfs)
        tensor = torch.from_numpy(waveform).to(self.device).unsqueeze(0)

        start_feature = time.perf_counter()
        features = self.extractor(tensor)
        feature_ms = (time.perf_counter() - start_feature) * 1000.0

        start_inference = time.perf_counter()
        logits = self.model(features)
        scores_tensor = torch.sigmoid(logits)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - start_inference) * 1000.0

        scores = scores_tensor.detach().cpu().numpy().astype(np.float32)
        active_mask = scores >= self.thresholds
        active_labels = [name for name, active in zip(self.class_names, active_mask) if bool(active)]
        status = "known_labels_detected" if active_labels else "no_confident_known_class"
        score_map = {name: float(score) for name, score in zip(self.class_names, scores)}
        total_ms = (time.perf_counter() - start_total) * 1000.0
        return Prediction(
            active_labels=active_labels,
            scores=score_map,
            status=status,
            input_rms_dbfs=original_rms,
            feature_ms=feature_ms,
            inference_ms=inference_ms,
            total_ms=total_ms,
        )

    def predict_file(self, path: str | Path) -> Prediction:
        waveform = load_audio(path, self.sample_rate)
        return self.predict_waveform(waveform)

    def predict_file_windows(self, path: str | Path, hop_seconds: float | None = None) -> list[dict]:
        """Run window-level inference across an entire file.

        Returns one record per window with time boundaries and prediction details.
        The last partial window is zero-padded so short tail events are not silently discarded.
        """
        waveform = load_audio(path, self.sample_rate)
        hop_seconds = float(hop_seconds if hop_seconds is not None else self.hop_seconds)
        if not np.isfinite(hop_seconds) or hop_seconds <= 0:
            raise ValueError("hop_seconds must be a positive finite number")
        if hop_seconds > self.window_seconds:
            raise ValueError("hop_seconds must not exceed window_seconds")

        records: list[dict] = []
        for window in iter_waveform_windows(
            waveform,
            sample_rate=self.sample_rate,
            window_seconds=self.window_seconds,
            hop_seconds=hop_seconds,
        ):
            prediction = self.predict_waveform(window.waveform)
            records.append(
                {
                    "window_index": window.index,
                    "start_seconds": window.start_seconds,
                    "end_seconds": window.valid_end_seconds,
                    **prediction.as_dict(),
                }
            )
        return records
