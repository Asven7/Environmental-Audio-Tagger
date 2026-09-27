#!/usr/bin/env python
from __future__ import annotations

import argparse
import queue
import sys
import time

import numpy as np

from esaudio.inference import AudioTagger
from esaudio.streaming import StreamingWindowBuffer


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous microphone inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        print(
            "sounddevice is not installed. Install the optional live dependency with: pip install '.[live]'",
            file=sys.stderr,
        )
        raise SystemExit(2)

    tagger = AudioTagger.from_files(args.checkpoint, args.thresholds, device=args.device)
    import torch
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hop_seconds = float(payload["project_config"].get("hop_seconds", 1.0))
    hop_samples = round(hop_seconds * tagger.sample_rate)
    buffer = StreamingWindowBuffer(tagger.window_samples, hop_samples)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)

    def callback(indata, frames, timing, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        mono = np.asarray(indata, dtype=np.float32).mean(axis=1)
        try:
            audio_queue.put_nowait(mono.copy())
        except queue.Full:
            print("Audio queue full; dropping chunk", file=sys.stderr)

    blocksize = min(hop_samples, round(0.1 * tagger.sample_rate))
    print("Starting microphone stream. Press Ctrl+C to stop.")
    print(f"sample_rate={tagger.sample_rate}, window={tagger.window_seconds}s, hop={hop_seconds}s")
    with sd.InputStream(
        samplerate=tagger.sample_rate,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    ):
        try:
            while True:
                chunk = audio_queue.get(timeout=1.0)
                for window in buffer.push(chunk):
                    result = tagger.predict_waveform(window.waveform)
                    print(
                        f"[{window.index:04d}] labels={result.active_labels or ['NO_CONFIDENT_KNOWN_CLASS']} "
                        f"compute={result.total_ms:.1f}ms"
                    )
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()
