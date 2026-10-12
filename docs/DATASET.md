# Dataset and split protocol

## Source and rights

The main research corpus is **UrbanSound8K**, obtained by the user and stored outside Git, conventionally at `data/UrbanSound8K/`. The repository does not redistribute the corpus; the dataset's own license and attribution requirements apply. See [References](REFERENCES.md).

Expected layout:

```text
data/UrbanSound8K/
  audio/fold1/ ... audio/fold10/
  metadata/UrbanSound8K.csv
```

## Classes

Known labels (eight outputs): `air_conditioner`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `jackhammer`, `siren`, `car_horn`.

For the limited rejection experiment, `gun_shot` and `street_music` are held out of training. They are evaluated only as examples outside the eight known target categories; this does not establish broad open-set generalization.

## Train, validation and test separation

- **Train:** official folds 1–7
- **Validation:** official fold 8
- **Test:** official folds 9–10

The split is defined **before** constructing synthetic mixtures. Each source is first associated with an occurrence identifier `(fsID, classID, occurrenceID)`. Where the broader Freesound recording identifier `fsID` spans the configured train/validation/test partitions, **all** its rows are excluded instead of being reassigned. The recorded metadata audit excluded `fsID` values `106905` and `180937` (102 clips in total).

The recorded single-source manifest counts, after the cross-split exclusion, were:

| Split | Known single clips | Held-out clips |
|---|---:|---:|
| Train | 5,209 | 0 |
| Validation | 673 | 130 |
| Test | 1,374 | 263 |

These counts come from the project's historical local audit. The full raw UrbanSound8K copy and generated research manifest files are not included in this public Git checkout.

## Controlled two-source mixtures

The model's multi-label examples are primarily derived from mixtures of two **different known classes** selected within the same split. Mix construction uses the following frozen factors:

- Relative first-versus-second source level: **−6, 0, +6 dB**.
- Nominal portion of the second source present in the output window: **0.25, 0.50, 1.00**.
- Generated recipe counts: **4,000 train**, **600 validation**, **1,200 test**.

Mixture metadata records both source paths and groups, label indices/names, relative level, overlap ratio and split. Recipes balance unordered class pairs and experimental conditions; two source events from the same broader `fsID` are not paired when that identity is available. The audio is synthesized on demand rather than committing derived mixture WAV files.

**Interpretation:** UrbanSound8K is fundamentally a single-label dataset, and combining recordings is not equivalent to annotating naturally overlapping real-world scenes. A label identifies a source recording's annotation; it does not guarantee precise event presence throughout every two-second crop.

## Generating and checking manifests

```powershell
python scripts\inspect_urbansound8k.py --dataset-root data\UrbanSound8K
python scripts\prepare_urbansound8k.py --dataset-root data\UrbanSound8K --output-dir artifacts\manifests_phase5 --include-mixtures
python scripts\inspect_mixture_manifests.py --manifest-dir artifacts\manifests_phase5
```

These commands use the current configuration and local dataset. **Generating new artifacts is not a license to revise frozen reported test results.** See [Methods](METHODS.md) for how the dataset enters training and [Results](RESULTS.md) for the already recorded outcomes.
