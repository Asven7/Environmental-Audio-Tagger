#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from esaudio.config import load_config
from esaudio.demo_data import generate_demo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic audio data for smoke tests and demos")
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="sample_data/demo")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = generate_demo_dataset(args.output, config)
    print(json.dumps(paths, indent=2))
    print("NOTE: this synthetic dataset is only for engineering verification, not research claims.")


if __name__ == "__main__":
    main()
