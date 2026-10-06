#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from esaudio.deployment import select_frozen_deployment
from esaudio.inference import AudioTagger
from esaudio.streaming import analyze_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test Phase-13 sequential file windowing."
    )
    parser.add_argument("--audio", required=True)
    parser.add_argument(
        "--experiment-root",
        default="artifacts/experiments_phase11",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument("--max-print", type=int, default=5)
    args = parser.parse_args()

    selection = select_frozen_deployment(
        args.experiment_root,
        model_name="crnn",
    )
    tagger = AudioTagger.from_files(
        selection.checkpoint,
        selection.thresholds,
        device=args.device,
    )
    results = analyze_file(tagger, args.audio)

    print("=== Phase 13A File-Window Smoke ===")
    print(
        f"deployment={selection.model_name} seed={selection.seed} "
        f"device={args.device}"
    )
    print(
        f"window_seconds={tagger.window_seconds} "
        f"hop_seconds={tagger.hop_seconds}"
    )
    print(f"windows={len(results)}")
    for result in results[: max(0, args.max_print)]:
        payload = result.as_dict()
        payload["scores"] = {
            name: round(score, 4)
            for name, score in payload["scores"].items()
        }
        print(json.dumps(payload, ensure_ascii=False))

    print("Sequential file inference: PASSED")


if __name__ == "__main__":
    main()
