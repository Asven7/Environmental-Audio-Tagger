#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import queue
from pathlib import Path

import torch

from esaudio.deployment import select_frozen_deployment
from esaudio.inference import AudioTagger
from esaudio.streaming import StreamingInferenceEngine


def _resolve_device(requested: str) -> str:
    requested = str(requested).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("--device must be auto, cpu, or cuda")
    return requested


def _resolve_artifacts(args) -> tuple[Path, Path, str]:
    if bool(args.checkpoint) != bool(args.thresholds):
        raise ValueError("--checkpoint and --thresholds must be supplied together")
    if args.checkpoint:
        return Path(args.checkpoint), Path(args.thresholds), "explicit"

    selection = select_frozen_deployment(
        args.experiment_root,
        model_name="crnn",
    )
    return (
        Path(selection.checkpoint),
        Path(selection.thresholds),
        (
            f"crnn_seed{selection.seed} "
            f"validation_mAP={selection.best_validation_mAP:.6f}"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Local sounddevice microphone inference using the frozen rolling "
            "2-second window / 1-second hop protocol"
        )
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--thresholds")
    parser.add_argument(
        "--experiment-root",
        default="artifacts/experiments_phase11",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--block-seconds",
        type=float,
        default=0.1,
        help="Audio callback block duration; inference hop remains frozen.",
    )
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is required for local microphone capture. "
            "Install it with: python -m pip install sounddevice"
        ) from exc

    if args.list_devices:
        print(sd.query_devices())
        return

    checkpoint, thresholds, deployment = _resolve_artifacts(args)
    device = _resolve_device(args.device)
    tagger = AudioTagger.from_files(
        checkpoint,
        thresholds,
        device=device,
    )
    engine = StreamingInferenceEngine(tagger)

    block_seconds = float(args.block_seconds)
    if block_seconds <= 0:
        raise ValueError("--block-seconds must be positive")
    blocksize = max(1, round(tagger.sample_rate * block_seconds))

    chunks: queue.Queue = queue.Queue(maxsize=64)
    callback_status_messages: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        del frames, time_info
        if status:
            try:
                callback_status_messages.put_nowait(str(status))
            except queue.Full:
                pass
        try:
            chunks.put_nowait(indata.copy())
        except queue.Full:
            # Do not block PortAudio's real-time callback.
            try:
                callback_status_messages.put_nowait(
                    "audio queue full: input chunk dropped"
                )
            except queue.Full:
                pass

    print("=== Live Environmental Audio Tagger ===")
    print(f"deployment={deployment}")
    print(f"device={device}")
    print(f"sample_rate={tagger.sample_rate}")
    print(
        f"window_seconds={tagger.window_seconds} "
        f"hop_seconds={tagger.hop_seconds}"
    )
    print("Press Ctrl+C to stop.")

    stream_kwargs = {
        "samplerate": tagger.sample_rate,
        "channels": 1,
        "dtype": "float32",
        "blocksize": blocksize,
        "callback": callback,
    }
    if args.input_device is not None:
        stream_kwargs["device"] = args.input_device

    try:
        with sd.InputStream(**stream_kwargs):
            while True:
                chunk = chunks.get()
                while not callback_status_messages.empty():
                    print(
                        "AUDIO_STATUS:",
                        callback_status_messages.get_nowait(),
                    )

                predictions = engine.push(chunk)
                for prediction in predictions:
                    payload = prediction.as_dict()
                    payload["scores"] = {
                        name: round(float(score), 4)
                        for name, score in payload["scores"].items()
                    }
                    payload["processing_ms"] = round(
                        float(payload["processing_ms"]),
                        3,
                    )
                    print(json.dumps(payload, ensure_ascii=False))
    except KeyboardInterrupt:
        print("\nMicrophone capture stopped.")


if __name__ == "__main__":
    main()
