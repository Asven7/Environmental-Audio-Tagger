# Phase 12 — Frozen Inference and Runtime Benchmark Protocol

## Status

**IMPLEMENTED — WAITING FOR USER LOCAL VERIFICATION**

## Proposal contract

The real-time design uses a two-second analysis window and a one-second hop. After the
first complete window is available, the system should update once per hop. The compute
time for a complete window must remain below the one-second hop so processing does not
accumulate backlog.

Runtime reporting separates:

- compute time from an already available complete waveform window to final output;
- update period (the one-second hop);
- initial latency (at least the two-second window formation time plus compute);
- Log-Mel feature extraction time;
- model inference time.

The target operating point is batch size one on an ordinary personal computer, not an
industrial low-latency guarantee.

## Audit of the pre-Phase-12 implementation

The repository already had a useful `AudioTagger` checkpoint/threshold inference path and
a legacy runtime script. The old `scripts/benchmark_runtime.py` contained a hard-coded
1000 ms stream-hop expression rather than deriving the hop from the checkpoint. The demo
pipeline separately used `tagger.hop_seconds`, which is the correct source.

Phase 12 therefore adds a canonical frozen runtime protocol rather than using the legacy
script for final runtime claims.

## Deployment-run selection

Runtime/demo work needs one concrete CRNN artifact.

The deployment seed is selected using **validation performance only**:

```text
model family: CRNN
criterion: highest best_validation_mAP in training_index.json
tie break: lower seed
eligible seeds: 13, 23, 37
```

For the completed Phase-11 experiment this selects:

```text
CRNN seed 23
best validation mAP = 0.675714...
```

No held-out test metric is read by the deployment selector. The selected checkpoint and
threshold file must still match their SHA-256 values in `experiment_freeze.json`.

This choice does not change Phase-11 results and does not retune the model.

## Canonical runtime measurement

`scripts/benchmark_frozen_runtime.py` uses batch size one and reports two timing views.

### Canonical total compute

The primary real-time number times the public:

```text
AudioTagger.predict_waveform(...)
```

path.

CUDA is synchronized before and after the timed call so GPU kernels are not accidentally
reported as asynchronous near-zero host time.

This `canonical_total.p95_ms` is used for the real-time no-backlog criterion:

```text
canonical_total.p95_ms < hop_seconds * 1000
```

### Stage breakdown

An equivalent staged path reports:

```text
preprocess
host_to_device
feature
model
postprocess
staged_total
```

Before benchmarking, its class-score vector must match the canonical AudioTagger output
within a small numeric tolerance.

## Timed-region boundary

The timed compute region begins when a complete mono waveform is already available in
host memory.

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
microphone capture time
disk I/O
audio-file decode
resampling used only to obtain the representative benchmark waveform
```

Those exclusions are intentional. Microphone capture contributes to the initial
window-formation latency, while disk I/O is not part of live microphone inference.

## Representative input

The benchmark script deterministically takes the first single-event item from
`known_val.csv`, loads/resamples it once, and then performs timed iterations on that
in-memory waveform.

Validation data are used only as a representative signal shape/content source; runtime
measurement does not calculate labels, accuracy, F1, mAP, or any held-out test metric.

## Reported values

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

The report also stores:

```text
batch_size = 1
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

## CPU and CUDA

The official local verification should benchmark both:

```text
CPU
CUDA
```

The project must retain a CPU inference path. CUDA is an acceleration path, not a
requirement for scientific correctness.

Runtime numbers are hardware-specific and should always be reported together with the
machine/device information.

## Warm-up and sample count

Official Phase-12 local benchmark:

```text
warm-up iterations: 10
measured iterations: 100
batch size: 1
primary statistic: p95 total compute
```

Do not infer runtime performance from training duration.

## Scientific boundary

Phase 12 does not:

- retrain;
- retune thresholds;
- recompute held-out accuracy;
- choose a seed using held-out test performance;
- alter Phase-11 frozen results.

It only verifies the frozen inference path and measures engineering latency.
