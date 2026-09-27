# Phase 3 — Configuration and Audio Foundation

## Goal

Establish a strict, testable foundation for all later dataset, feature, training, and inference phases.

This phase covers only:

- YAML configuration loading and validation;
- conversion of configured window/hop durations to sample counts;
- audio file loading;
- stereo-to-mono conversion;
- resampling;
- deterministic crop/pad behavior;
- bounded RMS normalization with silence protection;
- controlled two-source waveform mixing;
- audio save/load round-trip behavior.

Feature extraction, dataset manifests, augmentation orchestration, and model training are intentionally deferred to later phases.

## Proposal alignment

The approved proposal requires mono conversion, a common sampling rate, fixed two-second windows with a one-second hop, a consistent amplitude-scaling policy, crop/padding for file input, and controlled two-source mixtures. The code in this phase provides the reusable primitives for those requirements.

## Key design decisions

### Configuration is validated early

Invalid experimental settings should fail before training starts. Validation includes:

- positive sample rate and durations;
- hop duration not larger than the window;
- disjoint target and held-out classes;
- disjoint train/validation/test folds;
- valid feature frequency range below Nyquist;
- valid model/training numeric ranges;
- overlap ratios in `(0, 1]`;
- decision thresholds in `(0, 1)`.

### Mono conversion uses channel averaging

Stereo/multi-channel input is averaged across channels. This is deterministic and avoids the amplitude doubling that channel summation can cause.

### Resampling uses polyphase filtering

`scipy.signal.resample_poly` is used because it is mature, local, deterministic, and sufficient for this project. No external audio service is required.

### Short clips are right-padded

A shorter input keeps its original start time and receives zero padding at the end. This is simple and matches file/stream window semantics.

### Long clips support explicit crop policies

- `start`: first samples;
- `center`: deterministic center crop for evaluation;
- `energy`: select the highest-energy fixed-length region, useful for reducing the risk of cutting away the labeled event during dataset preparation.

### RMS normalization is bounded

Low-energy windows are not aggressively amplified. A gain clamp and silence threshold reduce the risk of turning background noise into a large signal.

### Mixture level convention

`relative_db = first source level - second source level`.

Therefore:

- `+6 dB`: first source is stronger;
- `0 dB`: equal nominal RMS level;
- `-6 dB`: first source is weaker.

If the final mixture would clip, the complete mixture is scaled down together so the source ratio is preserved.


### Train-time global time-shift primitive

The dataset augmenter depends on `apply_global_time_shift`. It shifts the entire
waveform inside the fixed window with zero fill and no circular wrap-around. This
keeps augmentation semantics physically simple: audio moved outside the window is
discarded instead of reappearing at the opposite edge.

## Acceptance criteria

Local verification on the target Windows laptop completed successfully on 2026-09-27. The phase-focused suites and the full regression suite passed after restoring the dataset augmentation time-shift compatibility helper.

| ID | Requirement | Test | Status |
|---|---|---|---|
| R-01 | Config loads and rejects invalid reproducibility settings | `tests/test_config.py` | USER VERIFIED |
| R-02 | Audio files load as finite mono float32 | `test_load_audio_downmixes_and_resamples` | USER VERIFIED |
| R-03 | Arbitrary input sample rate is converted to configured rate | audio I/O tests | USER VERIFIED |
| R-04 | Fixed window size is produced by crop/pad | `tests/test_audio.py` | USER VERIFIED |
| R-05 | Evaluation crop behavior is deterministic | center crop test | USER VERIFIED |
| R-06 | Energy crop selects a high-energy region | energy crop test | USER VERIFIED |
| R-07 | Silence is not amplified by normalization | silence test | USER VERIFIED |
| R-08 | Controlled mixture preserves requested level convention | spectral level test | USER VERIFIED |
| R-09 | Invalid overlap/config values fail clearly | validation tests | USER VERIFIED |
| R-10 | Global train-time shift preserves length and never wraps samples | time-shift unit tests | USER VERIFIED |

### Local verification evidence

```text
tests/test_config.py       8 passed
tests/test_audio.py       13 passed
tests/test_audio_io.py     4 passed
tests/test_demo_dataset.py 1 passed
full regression           31 passed
```

The full regression suite is the phase gate because it also verifies compatibility with previously implemented dataset code.
