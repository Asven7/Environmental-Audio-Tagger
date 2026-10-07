# Phase 12 — Frozen Inference and Runtime Benchmark Protocol

## Status

**USER VERIFIED — PROTOCOL AND CPU/CUDA BENCHMARK COMPLETED**

The authoritative measured values are in [`PHASE_12_RUNTIME_RESULTS.md`](PHASE_12_RUNTIME_RESULTS.md).

## Proposal contract

The real-time design uses:

```text
analysis window = 2.0 s
stream hop = 1.0 s
batch size = 1
```

After the first complete window is available, compute for one update must remain below the one-second hop so processing does not accumulate backlog.

Runtime reporting separates:

- compute time from an already available complete waveform window to final output;
- update period;
- initial latency caused by collecting the first window;
- Log-Mel feature time;
- model inference time.

The result is a hardware-specific engineering benchmark, not an industrial latency guarantee.

## Pre-Phase-12 audit

The repository already had `AudioTagger` inference and a legacy runtime script. Phase 12 established the canonical frozen runtime path and required the hop to come from the frozen checkpoint/configuration rather than a hard-coded timing assumption.

## Deployment-run selection

Runtime/demo work requires one concrete CRNN artifact.

Selection rule:

```text
model family = CRNN
criterion = highest best_validation_mAP
eligible seeds = 13, 23, 37
tie break = lower seed
```

Completed selection:

```text
CRNN seed 23
best validation mAP = 0.6757137110147023
```

No held-out test metric is read by the deployment selector.

The selected checkpoint/threshold artifacts must match the hashes in `experiment_freeze.json`.

## Canonical runtime measurement

`scripts/benchmark_frozen_runtime.py` uses batch size one and reports two views.

### Canonical total compute

Primary path:

```text
AudioTagger.predict_waveform(...)
```

CUDA is synchronized around the timed call.

No-backlog criterion:

```text
canonical_total.p95_ms < hop_seconds * 1000
```

### Stage breakdown

Equivalent staged timing reports:

```text
preprocess
host_to_device
feature
model
postprocess
staged_total
```

Its score vector must match the canonical output within numeric tolerance before benchmarking.

## Timed-region boundary

Included:

```text
fixed-window preparation
RMS normalization
host-to-device transfer
Log-Mel extraction
CRNN forward
sigmoid / CPU score transfer / threshold decision
```

Excluded:

```text
microphone capture
disk I/O
audio-file decode
representative-input resampling outside the timed region
```

## Representative input

The benchmark deterministically uses a known validation single-event sample only as an in-memory representative waveform.

No labels, accuracy, F1, mAP, or held-out test metric are calculated by runtime measurement.

## Reported statistics

For each timing category:

```text
n
mean
sample standard deviation
min
p50
p95
p99
max
```

Report metadata includes:

```text
batch_size
window_ms
stream_hop_ms
p95_compute_to_hop_ratio
p95_no_backlog_margin_ms
meets_no_backlog_criterion_p95
initial_prediction_latency_p95_ms
checkpoint SHA-256
threshold SHA-256
deployment selection provenance
hardware/software environment
representative input provenance
```

## Official verification protocol

```text
CPU and CUDA
10 warm-up iterations
100 measured iterations
batch size 1
primary statistic = p95 total compute
```

## Verified result

```text
CPU canonical p95  = 4.358 ms
CUDA canonical p95 = 1.561 ms
hop                 = 1000 ms

CPU  no-backlog = PASS
CUDA no-backlog = PASS
```

Initial prediction latency remains approximately two seconds plus compute time because the first analysis window must first be collected.

## Scientific boundary

Phase 12 did not:

- retrain;
- retune thresholds;
- recompute held-out accuracy;
- choose a seed using held-out test performance;
- alter Phase-11 frozen results.

It measures engineering latency of the already frozen deployment.
