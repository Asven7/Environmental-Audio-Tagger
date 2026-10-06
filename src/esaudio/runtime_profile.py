from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .audio import load_audio, normalize_rms, pad_or_crop
from .experiments import file_sha256
from .inference import AudioTagger


def resolve_runtime_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Runtime benchmark supports only CPU or CUDA")
    return device


def _stats(values_ms: list[float]) -> dict[str, float | int]:
    values = np.asarray(values_ms, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty timing list")
    if not np.isfinite(values).all():
        raise ValueError("Timing values must be finite")
    return {
        "n": int(values.size),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min_ms": float(np.min(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
    }


def representative_waveform_from_manifest(
    manifest_path: str | Path,
    audio_root: str | Path,
    sample_rate: int,
) -> tuple[np.ndarray, dict]:
    """Load one deterministic known validation single-event waveform.

    File I/O and resampling occur before timed runtime iterations. The timed
    contract begins once a complete host-memory waveform is available.
    """

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Runtime input manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path)
    if frame.empty:
        raise ValueError("Runtime input manifest is empty")
    if "sample_type" in frame.columns:
        singles = frame[frame["sample_type"].astype(str) == "single"]
        if not singles.empty:
            frame = singles
    row = frame.iloc[0]

    relative = Path(str(row["source_a"]))
    path = Path(audio_root) / relative
    waveform = load_audio(path, int(sample_rate))
    metadata = {
        "sample_id": str(row.get("sample_id", relative.as_posix())),
        "source_a": relative.as_posix(),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "audio_loading_in_timed_region": False,
    }
    return waveform, metadata


@dataclass
class StagedOutput:
    scores: dict[str, float]
    active_labels: list[str]
    status: str
    timings_ms: dict[str, float]


class RuntimeProfiler:
    """Batch-size-one profiler for the frozen AudioTagger path.

    ``canonical_total`` times the public ``AudioTagger.predict_waveform`` call
    and is the value used for the no-backlog real-time criterion. A second,
    equivalent staged path measures preprocessing, host-to-device transfer,
    Log-Mel feature extraction, model inference, and postprocessing separately.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        thresholds_path: str | Path,
        *,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.thresholds_path = Path(thresholds_path)
        self.device = resolve_runtime_device(device)

        self.tagger = AudioTagger.from_files(
            self.checkpoint_path,
            self.thresholds_path,
            device=str(self.device),
        )
        self.model = self.tagger.model
        self.extractor = self.tagger.extractor
        self.class_names = list(self.tagger.class_names)
        self.thresholds = np.asarray(self.tagger.thresholds, dtype=np.float32)

        payload = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        project = dict(payload["project_config"])
        self.model_spec = dict(payload["model_spec"])
        self.training_metadata = dict(payload.get("training_metadata", {}))

        self.sample_rate = int(project["sample_rate"])
        self.window_seconds = float(project["window_seconds"])
        self.hop_seconds = float(project["hop_seconds"])
        self.target_rms_dbfs = float(project.get("target_rms_dbfs", -20.0))
        self.window_samples = round(self.sample_rate * self.window_seconds)

        if self.sample_rate <= 0:
            raise ValueError("Checkpoint sample_rate must be positive")
        if self.window_seconds <= 0 or self.hop_seconds <= 0:
            raise ValueError("Checkpoint window/hop durations must be positive")
        if self.window_samples <= 0:
            raise ValueError("Checkpoint window length is invalid")
        if self.thresholds.shape != (len(self.class_names),):
            raise ValueError("Threshold count does not match checkpoint class count")

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _prepare_window(self, waveform: np.ndarray) -> np.ndarray:
        array = np.asarray(waveform, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(
                f"Runtime waveform must be mono [T], got shape {array.shape}"
            )
        if array.size == 0:
            raise ValueError("Runtime waveform must not be empty")
        if not np.isfinite(array).all():
            raise ValueError("Runtime waveform contains non-finite values")

        fixed = pad_or_crop(array, self.window_samples, strategy="center")
        fixed = normalize_rms(fixed, target_dbfs=self.target_rms_dbfs)
        return fixed.astype(np.float32, copy=False)

    @torch.inference_mode()
    def staged_predict(self, waveform: np.ndarray) -> StagedOutput:
        total_start = time.perf_counter()

        start = time.perf_counter()
        prepared = self._prepare_window(waveform)
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        self._sync()
        start = time.perf_counter()
        tensor = torch.from_numpy(prepared).unsqueeze(0).to(self.device)
        self._sync()
        transfer_ms = (time.perf_counter() - start) * 1000.0

        self._sync()
        start = time.perf_counter()
        features = self.extractor(tensor)
        self._sync()
        feature_ms = (time.perf_counter() - start) * 1000.0

        self._sync()
        start = time.perf_counter()
        logits = self.model(features)
        self._sync()
        model_ms = (time.perf_counter() - start) * 1000.0

        self._sync()
        start = time.perf_counter()
        score_array = (
            torch.sigmoid(logits)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        active_mask = score_array >= self.thresholds
        active_labels = [
            name
            for name, active in zip(self.class_names, active_mask)
            if bool(active)
        ]
        scores = {
            name: float(score)
            for name, score in zip(self.class_names, score_array)
        }
        status = (
            "known_labels_detected"
            if active_labels
            else "no_confident_known_class"
        )
        postprocess_ms = (time.perf_counter() - start) * 1000.0
        self._sync()

        staged_total_ms = (time.perf_counter() - total_start) * 1000.0
        return StagedOutput(
            scores=scores,
            active_labels=active_labels,
            status=status,
            timings_ms={
                "preprocess_ms": preprocess_ms,
                "host_to_device_ms": transfer_ms,
                "feature_ms": feature_ms,
                "model_ms": model_ms,
                "postprocess_ms": postprocess_ms,
                "staged_total_ms": staged_total_ms,
            },
        )

    def canonical_predict_timed(self, waveform: np.ndarray):
        self._sync()
        start = time.perf_counter()
        result = self.tagger.predict_waveform(waveform)
        self._sync()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return result, elapsed_ms

    def parity_check(self, waveform: np.ndarray, *, atol: float = 1e-5) -> dict:
        canonical, _ = self.canonical_predict_timed(waveform)
        staged = self.staged_predict(waveform)

        canonical_scores = {
            str(name): float(value)
            for name, value in canonical.scores.items()
        }
        if list(canonical_scores) != self.class_names:
            # Dict order is expected to be checkpoint class order. Compare by
            # class name below regardless, but record the difference.
            order_match = False
        else:
            order_match = True

        differences = [
            abs(canonical_scores[name] - staged.scores[name])
            for name in self.class_names
        ]
        max_abs_diff = max(differences, default=0.0)
        if max_abs_diff > float(atol):
            raise RuntimeError(
                "Staged runtime path does not match canonical AudioTagger scores: "
                f"max_abs_diff={max_abs_diff:.8g}"
            )
        return {
            "passed": True,
            "atol": float(atol),
            "max_abs_score_diff": float(max_abs_diff),
            "class_order_match": bool(order_match),
        }

    def benchmark(
        self,
        waveform: np.ndarray,
        *,
        warmup: int = 10,
        iterations: int = 100,
    ) -> dict:
        warmup = int(warmup)
        iterations = int(iterations)
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        if iterations < 2:
            raise ValueError("iterations must be at least 2")

        parity = self.parity_check(waveform)

        for _ in range(warmup):
            self.canonical_predict_timed(waveform)
            self.staged_predict(waveform)

        canonical_total: list[float] = []
        stage_values: dict[str, list[float]] = {
            "preprocess": [],
            "host_to_device": [],
            "feature": [],
            "model": [],
            "postprocess": [],
            "staged_total": [],
        }

        for _ in range(iterations):
            _, elapsed_ms = self.canonical_predict_timed(waveform)
            canonical_total.append(elapsed_ms)

            staged = self.staged_predict(waveform)
            timings = staged.timings_ms
            stage_values["preprocess"].append(timings["preprocess_ms"])
            stage_values["host_to_device"].append(
                timings["host_to_device_ms"]
            )
            stage_values["feature"].append(timings["feature_ms"])
            stage_values["model"].append(timings["model_ms"])
            stage_values["postprocess"].append(timings["postprocess_ms"])
            stage_values["staged_total"].append(timings["staged_total_ms"])

        canonical_stats = _stats(canonical_total)
        stream_hop_ms = self.hop_seconds * 1000.0
        window_ms = self.window_seconds * 1000.0
        p95_ms = float(canonical_stats["p95_ms"])

        return {
            "protocol_version": "batch1_runtime_profile_v1",
            "batch_size": 1,
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "window_ms": window_ms,
            "stream_hop_ms": stream_hop_ms,
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
            "timed_region": (
                "complete mono waveform already available in host memory -> "
                "preprocessing -> Log-Mel -> model -> sigmoid/threshold output"
            ),
            "excluded_from_timed_region": [
                "microphone capture time",
                "disk I/O",
                "audio-file decode/resample used to obtain representative waveform",
            ],
            "parity_with_canonical_inference": parity,
            "canonical_total": canonical_stats,
            "stages": {
                name: _stats(values)
                for name, values in stage_values.items()
            },
            "meets_no_backlog_criterion_p95": bool(p95_ms < stream_hop_ms),
            "p95_compute_to_hop_ratio": float(p95_ms / stream_hop_ms),
            "p95_no_backlog_margin_ms": float(stream_hop_ms - p95_ms),
            "initial_prediction_latency_p95_ms": float(window_ms + p95_ms),
            "checkpoint_sha256": file_sha256(self.checkpoint_path),
            "threshold_artifact_sha256": file_sha256(self.thresholds_path),
            "model_spec": self.model_spec,
            "training_metadata": self.training_metadata,
        }


def runtime_environment() -> dict:
    cuda_name = None
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(0)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": cuda_name,
        "torch_num_threads": int(torch.get_num_threads()),
    }


def save_runtime_report(
    report: dict,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Runtime report already exists: {path}; use explicit overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
