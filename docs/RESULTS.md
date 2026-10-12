# Recorded research and runtime results

These tables consolidate the **historical frozen reports** for the research configuration, rather than presenting a new rerun. Results refer to an UrbanSound8K-derived fixed-window tagging protocol with controlled mixtures; they are not performance measurements from the repository's small synthetic tone examples.

## Evaluation provenance

- Architecture × seed matrix: CNN and CRNN × `{13, 23, 37}` (six runs).
- Held-out known-set size: **2,574** windows/manifest examples; held-out-class examples: **263**.
- Mean ± **sample** standard deviation across the three seeds.
- Freeze identifier: `cnn_crnn_validation_freeze_v1`.
- Recorded freeze SHA-256: `0305a2aa821ed16a5f485cde1951d3720c47fa65dcc9d0ec9befd2e07f63be68`.
- Class thresholds were selected using validation examples, before the frozen test step.

These values are supported by the repository's historical reports, but the six original research checkpoint folders and raw per-run evaluation records are **not distributed** in the public checkout. For independent numerical verification, the original experimental artifact set must be supplied and checked by hash; see [Reproducibility](REPRODUCIBILITY.md).

## Held-out known-class results

| Metric | CNN, mean ± SD | CRNN, mean ± SD |
|---|---:|---:|
| mAP | 0.618628 ± 0.008281 | **0.727378 ± 0.007039** |
| F1 micro | 0.549715 ± 0.004061 | **0.592859 ± 0.010359** |
| F1 macro | 0.564576 ± 0.003953 | **0.617833 ± 0.013461** |
| Precision micro | 0.477551 ± 0.001776 | **0.520813 ± 0.018149** |
| Recall micro | 0.647589 ± 0.008033 | **0.688571 ± 0.010842** |
| Hamming loss | 0.194428 ± 0.000791 | **0.173433 ± 0.008472** |

CRNN mean mAP exceeds CNN by **0.108750 absolute points** under this specific evaluation protocol.

### Class-level F1 (three-seed mean)

| Known class | CNN | CRNN |
|---|---:|---:|
| air_conditioner | 0.401631 | **0.443732** |
| children_playing | 0.575626 | **0.583054** |
| dog_bark | 0.528906 | **0.661252** |
| drilling | 0.517914 | **0.574483** |
| engine_idling | 0.548679 | **0.618921** |
| jackhammer | **0.701231** | 0.687707 |
| siren | 0.578931 | **0.632981** |
| car_horn | 0.663692 | **0.740540** |

CRNN has a higher reported mean F1 for seven of eight classes. CNN is slightly higher for `jackhammer`.

## Single-source versus mixture results

| Subset | Model | Micro-F1 (mean ± SD) | Macro-F1 (mean ± SD) | mAP (mean ± SD) |
|---|---|---:|---:|---:|
| Single | CNN | 0.550776 ± 0.005033 | 0.586631 ± 0.005033 | 0.695764 ± 0.010346 |
| Mixture | CNN | 0.548797 ± 0.008190 | 0.551792 ± 0.007778 | 0.628511 ± 0.007605 |
| Single | CRNN | 0.614831 ± 0.018447 | 0.664000 ± 0.023633 | 0.822535 ± 0.011912 |
| Mixture | CRNN | 0.576376 ± 0.004244 | 0.582816 ± 0.008748 | 0.691822 ± 0.003890 |

Both architectures have lower recorded mAP on mixtures than on single-source clips.

### Controlled mixture factors: mean micro-F1

| Condition | CNN | CRNN |
|---|---:|---:|
| Overlap 0.25 | 0.528015 | 0.539953 |
| Overlap 0.50 | 0.552238 | 0.583650 |
| Overlap 1.00 | 0.565330 | 0.605508 |
| Relative level −6 dB | 0.553602 | 0.567973 |
| Relative level 0 dB | 0.559098 | 0.588966 |
| Relative level +6 dB | 0.533626 | 0.571980 |

The overlap pattern observed here reflects how the synthetic mixtures were constructed. Larger overlap is not necessarily easier in natural recordings.

## Held-out-class rejection: a major limitation

Rule: reject only if **no known-class score exceeds its validation-selected threshold**.

| Metric | CNN, mean ± SD | CRNN, mean ± SD |
|---|---:|---:|
| Known false rejection | 0.010231 ± 0.002864 | 0.004533 ± 0.001471 |
| Known acceptance | 0.989769 ± 0.002864 | 0.995467 ± 0.001471 |
| Held-out rejection | 0.012674 ± 0.007915 | **0.046895 ± 0.025317** |
| Held-out false acceptance | 0.987326 ± 0.007915 | **0.953105 ± 0.025317** |

For CRNN, approximately **95.31%** of the held-out examples are accepted as at least one known class. The project therefore **does not provide dependable open-set/OOD detection**. This negative result should remain visible in any academic report.

## Selected runtime and deployment

The recorded deployment within the chosen CRNN family is seed `23`, selected using **validation mAP = 0.6757137110147023**. The local research checkpoint and threshold file paths are:

```text
artifacts/experiments_phase11/crnn_seed23/best_model.pt
artifacts/experiments_phase11/crnn_seed23/thresholds.json
```

Runtime on the recorded Windows laptop (Intel Core i7-12700H, NVIDIA RTX 3050 Ti Laptop GPU) used batch size 1, 10 warm-up runs and 100 measured runs. Input was already decoded and in host memory: disk and audio capture times were excluded.

| Device | Mean canonical compute | p95 compute | 1-second-hop criterion |
|---|---:|---:|---|
| CPU | 3.435 ms | 4.358 ms | Met |
| CUDA | 1.424 ms | 1.561 ms | Met |

The first prediction still requires collecting the first **2-second** window. Neither timing measures total acquisition latency nor guarantees comparable speed on other machines.

## Limits of generalization

- Controlled overlaps are not independently annotated natural soundscapes.
- Clip-level source annotations can be imperfect for each two-second crop.
- Live-microphone accuracy has not been quantified using a separate annotated field dataset.
- Window tagging is not exact event localization, source counting, separation or recognition of arbitrary unseen classes.
- The reported six-run data are historical documented measurements; independent audit requires the locally stored original research artifacts.
