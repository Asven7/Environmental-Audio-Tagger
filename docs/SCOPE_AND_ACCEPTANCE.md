# Final Scope and Acceptance Criteria

## Recommendation Triage

The implementation deliberately applies only recommendations that materially improve correctness, reproducibility, or defense value.

| Recommendation | Decision | Implementation |
|---|---|---|
| Rename task conceptually from full SED to window-level tagging/event presence | MUST IMPLEMENT | README, docs, UI wording, inference semantics |
| Freeze target and held-out classes | MUST IMPLEMENT | `config/default.yaml` |
| Freeze train/validation/test folds | MUST IMPLEMENT | `config/default.yaml`, manifest builder |
| Prevent source leakage before mixing | MUST IMPLEMENT | `manifests.py::assert_no_source_leakage` |
| Formalize controlled mixture relative levels | MUST IMPLEMENT | manifest generation + grouped evaluation |
| Formalize temporal overlap | MUST IMPLEMENT | configurable overlap ratios + group metrics |
| Validation-only threshold selection | MUST IMPLEMENT | `evaluation.py`, `training.py` |
| Replace general “unknown” claim with limited rejection semantics | MUST IMPLEMENT | `no_confident_known_class` everywhere |
| Shared train/inference preprocessing | MUST IMPLEMENT | `audio.py`, `features.py` reused by both paths |
| CNN baseline vs CRNN comparison | MUST IMPLEMENT | `models.py`, experiment runner |
| Runtime measurement and no-backlog criterion | MUST IMPLEMENT | `runtime.py`, benchmark script |
| Multi-seed experiments | SHOULD IMPLEMENT | experiment runner + aggregator |
| Parameter-count reporting | SHOULD IMPLEMENT | training summary |
| Reference hardware documentation | SHOULD IMPLEMENT | runtime docs; user fills actual machine in final report |
| Small real/external validation set | SHOULD IMPLEMENT, data-dependent | pipeline supports file inference; no external dataset fabricated |
| Live microphone | SHOULD IMPLEMENT | live CLI implemented; hardware verification required locally |
| Gradio university-demo UI | SHOULD IMPLEMENT | implemented and build-verified |
| MFCC/RMS/ZCR auxiliary baseline | OPTIONAL/FUTURE | not implemented in MVP |
| LSTM comparison in addition to GRU | OPTIONAL/FUTURE | code supports LSTM by config, default uses GRU |
| Temporal score smoothing | OPTIONAL/FUTURE | not implemented |
| Calibration / advanced open-set methods | OPTIONAL/FUTURE | not implemented |
| Quantization / ONNX | OPTIONAL/FUTURE | not implemented |
| Database | SHOULD NOT IMPLEMENT | immutable experiment manifests are sufficient |
| REST API/backend service | SHOULD NOT IMPLEMENT | local scientific application; unnecessary layer |
| Authentication/authorization | SHOULD NOT IMPLEMENT | localhost application; no user accounts |
| Docker as required path | SHOULD NOT IMPLEMENT | audio/GPU/device mapping adds complexity; venv is simpler |
| Microservices, queues, Kubernetes | SHOULD NOT IMPLEMENT | unrelated to project objective |

## Core MVP

- Leakage-safe fixed data protocol.
- Fixed-window waveform preprocessing.
- Log-Mel features.
- CNN baseline.
- CNN+GRU CRNN.
- Multi-label training.
- Per-class validation thresholds.
- Frozen test evaluation.
- Controlled two-source mixtures.
- Relative-level and overlap analysis.
- File inference.
- Runtime benchmark.
- Automated tests.
- Reproducible configuration and documentation.

## Secondary Features

- Held-out class rejection analysis.
- Multi-seed experiment automation.
- Continuous microphone CLI.
- Gradio upload/microphone-recording UI.
- Synthetic demo dataset/checkpoints.

## Optional/Future Work

- Real multi-label annotated soundscape test set.
- LSTM/GRU direct comparison.
- Additional pretrained baseline.
- Calibration, OSR methods, ONNX, quantization.
- Advanced streaming UI.

## Explicitly Out of Scope

- Full SED onset/offset estimation.
- Source separation/counting/localization.
- General unknown-sound detection.
- Cloud production deployment.
- Multi-microphone arrays.
- Large foundation models.

## Acceptance Criteria

### Data Protocol

**Input:** UrbanSound8K root with official metadata.

**Expected behavior:**
- assign folds 1–7 to train, 8 to validation, 9–10 to test by default;
- include only configured target classes in known training data;
- reserve held-out classes from the main classifier;
- generate mixtures only within a split;
- detect source path leakage across known splits.

**Failure cases:** missing metadata, missing required metadata columns, impossible mixture generation, source leakage.

**Pass criterion:** manifest generation completes and `assert_no_source_leakage` raises no error.

### Preprocessing

**Input:** mono or multi-channel audio at an arbitrary sample rate.

**Expected output:** fixed-length mono float waveform at configured sample rate; same normalization policy in training and inference.

**Pass criterion:** unit tests for crop/pad, RMS handling, resampling path, and mixture safety pass.

### Feature Extraction

**Input:** `[B,T]` waveform tensor.

**Output:** `[B,1,n_mels,time_frames]` normalized Log-Mel tensor.

**Pass criterion:** deterministic shape and finite values on valid input.

### Models

**Input:** Log-Mel tensor.

**Output:** one logit per target class.

**Pass criterion:** CNN and CRNN forward tests pass; checkpoint can round-trip.

### Training

**Expected behavior:** optimize `BCEWithLogitsLoss`, select best checkpoint on validation mAP, early-stop when configured, never use test data.

**Pass criterion:** a smoke dataset run produces a checkpoint, history, threshold file, and training summary.

### Thresholding

**Expected behavior:** independently choose each class threshold from a fixed grid using validation F1.

**Pass criterion:** known deterministic fixture selects thresholds that recover expected labels.

### Evaluation

**Expected output:** micro/macro Precision/Recall/F1, mAP, Hamming Loss, per-class metrics, grouped controlled-condition metrics, and optional rejection metrics.

**Pass criterion:** evaluator runs from saved checkpoint + threshold file without retraining.

### File Inference

**Input:** audio file.

**Output:** one prediction record per fixed window, including times, label scores, active labels, and compute time.

**Pass criterion:** demo file produces records without exception.

### Rejection Semantics

**Expected behavior:** if no known class exceeds its threshold, return `no_confident_known_class`.

**Validation rule:** never describe that state as proof that an arbitrary unknown sound was detected.

### Runtime

**Input:** representative test windows, batch size 1.

**Output:** feature, inference, total mean/median/p95 latency.

**Pass criterion:** on the chosen reference machine, `total_ms_p95 < hop_seconds * 1000` for the final model.

### Live Microphone

**Input:** local audio device.

**Expected behavior:** emit overlapping windows using the tested stream buffer and print predictions without audio backlog.

**Failure cases:** missing `sounddevice`, missing PortAudio, denied microphone permissions, unsupported audio device.

**Pass criterion:** must be verified on the actual demonstration machine; it cannot be certified in a headless build environment without an audio device.
