# Phase 5 — Controlled Multi-Label Mixture Protocol

## Goal

Create deterministic two-label mixture metadata **after** the Phase 4 train/validation/test protocol is frozen. No Log-Mel extraction or model training is performed in this phase.

## Frozen inputs

Known classes remain the eight Phase 4 target classes. Held-out OOD classes (`gun_shot`, `street_music`) are never used as mixture sources for classifier training/evaluation mixtures.

The Phase 4 source-isolation policy remains active: the two broader recordings that cross configured splits are excluded before mixture generation, and all mixture sources are selected only from retained known singles within the same split.

## Default experimental grid

The default project configuration uses:

```text
relative dB:  -6, 0, +6
intersection: 25%, 50%, 100%
```

`relative_db` is defined as:

```text
level(source_a) - level(source_b)
```

Therefore `+6 dB` means source A is approximately 6 dB stronger than source B. `overlap_ratio` controls how much of source B is placed inside the output window. An overlap of `1.0` means full overlap; `0.5` places source B in the final half of the window.

The configured mixture counts are:

| Split | Mixtures |
|---|---:|
| train | 4000 |
| validation | 600 |
| test | 1200 |

These values are read from `config/default.yaml`; they are not hard-coded in the manifest generator.

## Pairing constraints

Every mixture must satisfy all of the following:

1. exactly two known classes;
2. the classes are different;
3. both sources belong to the same configured split;
4. the exact unordered source-group pair is used at most once;
5. when an UrbanSound8K `fsID` can be recovered for both sources, the two sources may not come from the same broader recording;
6. mixture labels are exactly the union of the two source labels;
7. `label_indices` and `label_names` remain aligned with the frozen target-class order;
8. OOD sources are never mixed into the known-class experimental manifests.

## Balance controls

With eight known classes there are 28 unordered class pairs. The generator keeps class-pair counts within one sample of each other.

There are nine global experimental conditions (`3 relative dB × 3 overlap ratios`). Their counts are also kept within one sample of each other.

The assignment of a class to source A versus source B is balanced within each unordered class pair to within one sample. This is important because the sign of `relative_db` is defined relative to source A; without orientation balancing, a class could become systematically louder or quieter simply because of source ordering.

For the default counts, the expected balance ranges are:

| Split | Class-pair count range | dB/overlap condition count range |
|---|---:|---:|
| train | 142–143 | 444–445 |
| validation | 21–22 | 66–67 |
| test | 42–43 | 133–134 |

## Determinism

Mixture metadata is deterministic for a fixed single-source manifest, target-class order, configuration, and `training.data_seed`.

The three split-specific seeds are derived from the base data seed:

```text
train: base_seed + 0
val:   base_seed + 1
test:  base_seed + 2
```

Changing the requested mixture count may change the generated schedule and source selections, so the final configuration must be frozen before research results are produced.

## Waveform synthesis

The manifest generator does not write mixed WAV files. At dataset load time, the two referenced source clips are loaded and passed to `mix_two_sources`.

Waveform-level safeguards already verified in the audio foundation include:

- fixed output length;
- controlled signed relative dB;
- configurable temporal overlap;
- a shared final peak scaling step when needed to prevent clipping;
- preservation of the relative source level under the shared peak scaling.

This on-demand approach avoids storing thousands of redundant synthetic WAV files while keeping the mixture recipe explicit in CSV metadata.

## Validation and audit

Targeted automated tests verify deterministic generation, balance, label union, source-group uniqueness, same-recording avoidance, manifest-summary metadata, and waveform-level relative dB realization.

After generating real manifests with mixtures, run:

```powershell
python scripts/inspect_mixture_manifests.py --manifest-dir artifacts\manifests_phase5
```

The command verifies cross-split source isolation, configured mixture counts, class-pair balance, dB/overlap balance, source-label union, and absence of mixtures from OOD manifests. It loads metadata only and does not train a model.

## Phase boundary

Phase 5 ends after controlled mixture manifests are generated and audited on the real UrbanSound8K baseline. Log-Mel extraction begins in Phase 6.

## Real-audio smoke test

After metadata validation, synthesize a few real mixtures directly from UrbanSound8K to verify that manifest paths, preprocessing, overlap, dB controls, and clipping protection work together:

```powershell
python scripts/smoke_mixture_audio.py --dataset-root data\UrbanSound8K --manifest-dir artifacts\manifests_phase5 --split train --count 3
```

The smoke test writes no WAV files and performs no training. Every tested mixture must have the configured window length, finite samples, and a peak no greater than the project safety limit.


### Synthetic demo compatibility

The no-repeat source-pair rule is strict for research manifests carrying explicit `occurrence:fsID:classID:occurrenceID` identities. Tiny synthetic demo manifests use path-fallback identities and may contain fewer possible source pairs than the configured smoke-test mixture count; only those demo/path-fallback manifests may reuse a source pair. This exception is for engineering smoke tests only and is never used for UrbanSound8K research manifests or reported scientific results.

## Verified real-data completion baseline

Phase 5 was verified on the real local UrbanSound8K research manifests after the Phase 4 leakage-safe split protocol was frozen.

Verified automated-test baseline:

```text
tests/test_mixture_protocol.py: 9 passed
tests/test_demo_dataset.py:     1 passed
full regression suite:          48 passed
```

Verified mixture-manifest generation:

| Split | Known singles | Controlled mixtures | Class-pair range | dB/overlap range | Unique mixture pairs |
|---|---:|---:|---:|---:|---:|
| train | 5209 | 4000 | 142–143 | 444–445 | 4000 |
| validation | 673 | 600 | 21–22 | 66–67 | 600 |
| test | 1374 | 1200 | 42–43 | 133–134 | 1200 |

The real manifest audit reported `Source-group leakage across splits: PASSED` and `Controlled mixture protocol validation: PASSED`.

The real-audio smoke test used three train mixtures at 22.05 kHz with 44,100-sample windows and energy-guided cropping. All three synthesized successfully, remained finite, respected the peak-safety limit, and covered overlap ratios 1.0, 0.5, and 0.25. No WAV files were written and no model training was performed.

Phase 5 status: **USER VERIFIED**.
