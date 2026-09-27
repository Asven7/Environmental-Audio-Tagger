from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from esaudio.config import load_config
from esaudio.dataset import AudioManifestDataset
from esaudio.demo_data import generate_demo_dataset
from esaudio.manifests import assert_no_source_leakage


ROOT = Path(__file__).resolve().parents[1]


def test_generate_demo_dataset_and_load_batch(tmp_path):
    config = load_config(ROOT / "config" / "demo.yaml")
    paths = generate_demo_dataset(
        tmp_path,
        config,
        samples_per_class_per_split={"train": 2, "val": 1, "test": 1},
    )
    frames = {
        split: pd.read_csv(paths[f"known_{split}"])
        for split in ("train", "val", "test")
    }
    assert_no_source_leakage(frames)
    dataset = AudioManifestDataset(paths["known_train"], tmp_path, config, split="train")
    assert len(dataset) > 0
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["waveform"].shape[1] == 8000
    assert batch["target"].shape[1] == 4
