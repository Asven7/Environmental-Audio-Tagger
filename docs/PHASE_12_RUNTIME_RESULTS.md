# Phase 12 — Frozen Inference and Runtime Results

## Status

**USER VERIFIED**

## Deployment artifact

Deployment/model-family selection was performed using validation performance only.

```text
model = crnn
seed = 23
best_validation_mAP = 0.6757137110147023
checkpoint = artifacts/experiments_phase11/crnn_seed23/best_model.pt
thresholds = artifacts/experiments_phase11/crnn_seed23/thresholds.json
```

The selected checkpoint and threshold artifact were already part of the Phase-11 frozen
experiment matrix.

## Runtime protocol

Official local runtime protocol:

```text
batch size: 1
window: 2.0 s
hop: 1.0 s
warm-up iterations: 10
measured iterations: 100
primary criterion: canonical total p95 < 1000 ms
```

The canonical total timing measures the public `AudioTagger.predict_waveform(...)` path
from an already available mono waveform in host memory to final scores/threshold output.

Disk I/O and microphone capture are excluded from this compute timing.

## CPU result

```text
device = cpu
mean = 3.435 ms
p50 = 3.244 ms
p95 = 4.358 ms
p99 = 4.841 ms
max = 5.634 ms

hop = 1000.000 ms
p95 / hop = 0.004358
p95 margin = 995.642 ms
no-backlog criterion = PASS

initial prediction latency p95 ≈ 2004.358 ms
```

Stage timing:

```text
preprocess p95      = 0.185 ms
host-to-device p95  = 0.017 ms
feature p95         = 0.643 ms
model p95           = 3.201 ms
postprocess p95     = 0.044 ms
```

The model forward pass is the dominant CPU compute component.

## CUDA result

```text
device = cuda
mean = 1.424 ms
p50 = 1.414 ms
p95 = 1.561 ms
p99 = 1.722 ms
max = 1.725 ms

hop = 1000.000 ms
p95 / hop = 0.001561
p95 margin = 998.439 ms
no-backlog criterion = PASS

initial prediction latency p95 ≈ 2001.561 ms
```

Stage timing:

```text
preprocess p95      = 0.134 ms
host-to-device p95  = 0.092 ms
feature p95         = 0.433 ms
model p95           = 0.704 ms
postprocess p95     = 0.078 ms
```

The CUDA path reduces model-forward latency substantially while introducing the expected
small host-to-device overhead.

## CPU vs CUDA

Approximate speedups:

```text
canonical total p95: CPU / CUDA ≈ 2.79×
canonical mean:      CPU / CUDA ≈ 2.41×
model p95:           CPU / CUDA ≈ 4.55×
feature p95:         CPU / CUDA ≈ 1.48×
```

Both paths are far below the one-second hop budget.

```text
CPU p95 uses  ~0.44% of the hop budget
CUDA p95 uses ~0.16% of the hop budget
```

Therefore GPU acceleration is useful but is not required for the real-time feasibility
claim on the tested machine.

## Inference parity

For both official CPU and CUDA measurements:

```text
parity_max_abs_score_diff = 0
```

The staged runtime path therefore matched the canonical inference output for the
representative benchmark input.

## Interpretation of latency

The following quantities are intentionally separated:

```text
analysis window = 2 seconds
update period = 1 second
compute latency = a few milliseconds
initial prediction latency ≈ 2 seconds + compute latency
```

The system should not be described as having an end-to-end initial latency of only
1–4 ms. The first result requires a complete two-second analysis window.

After the first window is available, the compute stage is sufficiently fast to produce
updates every one-second hop without backlog.

## Real-time conclusion

The Phase-12 real-time engineering target is satisfied on both CPU and CUDA.

The scientifically appropriate claim is:

> The frozen CRNN inference pipeline, using batch size one, processes each two-second
> analysis window well within the one-second hop budget on the tested laptop, on both
> CPU and RTX 3050 Ti CUDA execution.

This is a hardware-specific engineering result and not a universal latency guarantee.

## Deployment recommendation

For subsequent UI/demo work:

```text
preferred device policy = auto
if CUDA is available -> CUDA
otherwise -> CPU fallback
```

Reasons:

- CUDA provides lower compute latency on the tested machine.
- CPU independently satisfies the real-time no-backlog criterion.
- Maintaining the CPU path improves portability and demo reliability.

Phase 13 should therefore not require a GPU to function correctly.

## Scientific boundary

Phase 12 did not:

- retrain either model;
- retune thresholds;
- recompute accuracy/test metrics;
- modify Phase-11 held-out results;
- select a seed using held-out test performance.

Runtime results are engineering measurements of the already frozen model.
