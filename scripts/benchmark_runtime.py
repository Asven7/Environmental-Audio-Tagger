#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from esaudio.inference import AudioTagger
from esaudio.runtime import benchmark_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark batch=1 runtime on prerecorded samples")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()

    tagger = AudioTagger.from_files(args.checkpoint, args.thresholds, device=args.device)
    result = benchmark_manifest(tagger, args.manifest, args.audio_root, args.max_samples)
    result["stream_hop_ms"] = tagger.window_seconds * 0 + 1000.0  # proposal default; overridden below when known
    # The checkpoint stores hop_seconds separately.
    import torch
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    result["stream_hop_ms"] = float(payload["project_config"].get("hop_seconds", 1.0)) * 1000.0
    result["meets_no_backlog_criterion_p95"] = bool(result["total_ms_p95"] < result["stream_hop_ms"])
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
