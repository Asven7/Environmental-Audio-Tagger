# Dataset Protocol

## Primary dataset

The main research dataset is UrbanSound8K. The raw dataset is stored locally under `data/UrbanSound8K/` and is never committed to Git.

Expected layout:

```text
data/UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── ...
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

## Frozen classes

Known target classes:

- `air_conditioner`
- `children_playing`
- `dog_bark`
- `drilling`
- `engine_idling`
- `jackhammer`
- `siren`
- `car_horn`

Held-out OOD classes:

- `gun_shot`
- `street_music`

The held-out classes are never used to train the eight-output known-class classifier. Their validation/test samples are used only for the limited rejection experiment.

## Frozen fold partition

- train: folds 1–7
- validation: fold 8
- test: folds 9–10

Official fold labels are preserved; clips are never reassigned to a different fold.

## Source identity and leakage policy

UrbanSound8K filenames encode `fsID-classID-occurrenceID-sliceID.wav`.

The hard occurrence identity used by the project is:

```text
(fsID, classID, occurrenceID)
```

All slices of one occurrence must remain in one official fold.

The real metadata also contains a very small number of broader `fsID` recordings whose different annotated events appear across the configured train/validation/test boundary. Because those clips originate from the same Freesound recording and may share background/acoustic conditions, the project applies a stricter research rule:

> Any `fsID` that spans more than one configured split is excluded from **all** research manifests.

This is an exclusion rule, not a reassignment rule. It strengthens source independence while preserving official fold membership for every retained clip.

The Phase 4 real-data audit found two cross-split `fsID` recordings (`106905`, `180937`). The code detects these dynamically from metadata and the configured folds.

## Single-source manifests

Phase 4 creates only single-source manifests. Mixtures are deliberately disabled until Phase 5.

Manifest columns include:

```text
sample_id
split
sample_type
source_a
source_b
source_group_a
source_group_b
label_indices
label_names
relative_db
overlap_ratio
is_ood
```

`source_group_a` uses the annotated-occurrence identity so overlapping slices from one event can be tracked as one source group.

## Reproducibility

Before manifest generation, run:

```powershell
python scripts/inspect_urbansound8k.py --dataset-root data\UrbanSound8K
```

Then generate Phase 4 manifests without mixtures:

```powershell
python scripts/prepare_urbansound8k.py --dataset-root data\UrbanSound8K --output-dir artifacts\manifests
```

Do **not** pass `--include-mixtures` during Phase 4.

The generated `manifest_summary.json` records the target/held-out classes, source-group unit, cross-split recording policy, excluded `fsID` values, excluded clip count, and per-split manifest counts.

## Verified Phase 4 manifest baseline

Real local verification produced the following leakage-safe single-source manifests:

| Split | Known singles | OOD singles |
|---|---:|---:|
| train | 5209 | 0 |
| validation | 673 | 130 |
| test | 1374 | 263 |

Two broader `fsID` recordings (`106905`, `180937`) crossed configured split boundaries and were excluded in full, removing 102 clips. Post-exclusion metadata totals are 6190 train, 803 validation, and 1637 test clips. No mixtures are included in the Phase 4 baseline.
