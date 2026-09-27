# Build Verification Record

This document records checks that were actually executed while constructing the repository. It is not a claim about UrbanSound8K research performance.

## Automated Tests

Command:

```bash
pytest
```

Result:

```text
13 passed
```

The exact captured output is stored in `artifacts/verification_pytest.txt`.

## Python Compilation Check

Command:

```bash
python -m compileall -q src scripts tests
```

Result: PASSED.

Captured in `artifacts/verification_compile.txt`.

## Synthetic End-to-End Training

Command:

```bash
python scripts/run_demo_pipeline.py
```

Verified:
- synthetic safe audio generation,
- manifest creation,
- CNN training,
- CRNN training,
- checkpoint creation,
- validation threshold creation,
- test evaluation,
- runtime benchmark.

The produced metrics are engineering smoke-test results only and must not be reported as environmental-audio research results.

## Saved-Checkpoint Inference

A synthetic test WAV was passed through the saved CRNN checkpoint and threshold file successfully.

Captured output: `artifacts/verification_inference.json`.

## Runtime Smoke Benchmark

The final synthetic CRNN benchmark used 8 prerecorded samples with batch size 1 on the build CPU.

At the final verification run:

- mean total compute time: approximately 2.87 ms,
- p95 total compute time: approximately 3.86 ms,
- demo stream hop: 500 ms,
- no-backlog criterion: PASSED.

See `artifacts/demo_crnn/runtime.json` for the exact machine-run output.

These numbers are not comparable to the final 22.05 kHz UrbanSound8K model until the real experiment is trained and benchmarked.

## UI

The Gradio `Blocks` interface was successfully constructed against the saved demo CRNN checkpoint. A local HTTP startup check also returned the Gradio HTML page during construction; physical browser/microphone behavior remains machine-dependent.

## Not Verified in This Environment

- Physical microphone capture, because no audio input device / `sounddevice` package was available in the headless build environment.
- Final UrbanSound8K training/evaluation, because the dataset was not supplied.
- Real-world multi-label soundscape accuracy, because no independently annotated real recording set was supplied.
