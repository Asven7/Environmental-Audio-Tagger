from pathlib import Path

import pandas as pd

from esaudio.config import load_config
from esaudio.manifests import build_urbansound8k_manifests


ROOT = Path(__file__).resolve().parents[1]


def test_urbansound_manifest_builder_with_fake_metadata(tmp_path):
    config = load_config(ROOT / "config" / "default.yaml")
    dataset_root = tmp_path / "UrbanSound8K"
    metadata_dir = dataset_root / "metadata"
    metadata_dir.mkdir(parents=True)
    rows = [
        {"slice_file_name": "a.wav", "fsID": 1, "fold": 1, "class": "dog_bark"},
        {"slice_file_name": "b.wav", "fsID": 2, "fold": 8, "class": "siren"},
        {"slice_file_name": "c.wav", "fsID": 3, "fold": 9, "class": "drilling"},
        {"slice_file_name": "d.wav", "fsID": 4, "fold": 8, "class": "gun_shot"},
        {"slice_file_name": "e.wav", "fsID": 5, "fold": 10, "class": "street_music"},
    ]
    pd.DataFrame(rows).to_csv(metadata_dir / "UrbanSound8K.csv", index=False)

    # Avoid requesting thousands of mixtures from the tiny fake metadata fixture.
    config["mixing"]["train_mixtures"] = 0
    config["mixing"]["val_mixtures"] = 0
    config["mixing"]["test_mixtures"] = 0
    paths = build_urbansound8k_manifests(dataset_root, tmp_path / "manifests", config)
    train = pd.read_csv(paths["known_train"])
    val = pd.read_csv(paths["known_val"])
    test = pd.read_csv(paths["known_test"])
    ood_val = pd.read_csv(paths["ood_val"])
    ood_test = pd.read_csv(paths["ood_test"])

    assert train.iloc[0]["source_a"] == "audio/fold1/a.wav"
    assert val.iloc[0]["source_a"] == "audio/fold8/b.wav"
    assert test.iloc[0]["source_a"] == "audio/fold9/c.wav"
    assert len(ood_val) == 1
    assert len(ood_test) == 1
