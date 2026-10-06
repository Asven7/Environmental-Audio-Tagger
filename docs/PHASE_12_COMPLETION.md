# Phase 12 Completion Record — Inference / Runtime Benchmarking

## Final status

**USER VERIFIED**

## Protocol-lock commit

```text
de1d845 feat: lock frozen inference runtime protocol
```

Repository state before official benchmark:

```text
clean
```

## Selected deployment run

```text
model = crnn
seed = 23
selection criterion = highest best_validation_mAP among frozen CRNN seeds
best_validation_mAP = 0.6757137110147023
```

Selection used validation information only.

## Official runtime measurements

### CPU

```text
batch_size = 1
canonical mean = 3.435 ms
canonical p95 = 4.358 ms
canonical p99 = 4.841 ms
max = 5.634 ms
hop = 1000 ms
no_backlog = true
p95 margin = 995.642 ms
```

### CUDA

```text
batch_size = 1
canonical mean = 1.424 ms
canonical p95 = 1.561 ms
canonical p99 = 1.722 ms
max = 1.725 ms
hop = 1000 ms
no_backlog = true
p95 margin = 998.439 ms
```

## Proposal requirement

The proposal requires per-window compute time to remain below the one-second hop.

Result:

```text
CPU:  PASS
CUDA: PASS
```

## Initial latency

Because the analysis window is two seconds:

```text
CPU initial prediction latency p95 ≈ 2004.358 ms
CUDA initial prediction latency p95 ≈ 2001.561 ms
```

The update period after the first window remains one second.

## Bottleneck

CPU:

```text
model p95 = 3.201 ms
feature p95 = 0.643 ms
```

CUDA:

```text
model p95 = 0.704 ms
feature p95 = 0.433 ms
```

The model forward pass is the largest measured compute component, especially on CPU.

## Inference parity

```text
CPU parity max absolute score difference = 0
CUDA parity max absolute score difference = 0
```

## Runtime artifacts

Git-ignored artifacts:

```text
artifacts/runtime_phase12/crnn_seed23_cpu.json
artifacts/runtime_phase12/crnn_seed23_cuda.json
```

## Verification

Before Phase 12 benchmark:

```text
runtime protocol tests = 6 passed
full repository tests = 128 passed
```

After benchmark:

```text
runtime report files = 2
Git working tree = clean
```

## Final Phase-12 conclusion

The frozen CRNN satisfies the project's real-time no-backlog requirement on the tested
hardware on both CPU and CUDA. CUDA is faster, but CPU performance is already comfortably
within the required one-second hop.

For Phase 13, use an automatic device policy with CPU fallback.

Phase 12 is complete. Phase 13 may now implement the file/microphone UI and streaming
buffer without changing the frozen model, thresholds, feature configuration, two-second
window, or one-second hop.
