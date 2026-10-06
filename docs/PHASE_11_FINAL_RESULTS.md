# Phase 11 — Final Multi-Seed Experimental Results

## Status
**USER VERIFIED**

Official matrix:
- CNN × seeds 13, 23, 37
- CRNN × seeds 13, 23, 37

Frozen held-out scope:
- known samples: 2574
- OOD samples: 263
- aggregation: mean ± sample standard deviation over 3 predeclared seeds
- retuning after held-out evaluation: none
- freeze version: `cnn_crnn_validation_freeze_v1`
- freeze SHA-256: `0305a2aa821ed16a5f485cde1951d3720c47fa65dcc9d0ec9befd2e07f63be68`

## Headline known-test results

| Metric | CNN | CRNN |
|---|---:|---:|
| mAP | 0.618628 ± 0.008281 | **0.727378 ± 0.007039** |
| F1 micro | 0.549715 ± 0.004061 | **0.592859 ± 0.010359** |
| F1 macro | 0.564576 ± 0.003953 | **0.617833 ± 0.013461** |
| Precision micro | 0.477551 ± 0.001776 | **0.520813 ± 0.018149** |
| Recall micro | 0.647589 ± 0.008033 | **0.688571 ± 0.010842** |
| Hamming loss | 0.194428 ± 0.000791 | **0.173433 ± 0.008472** |

CRNN absolute improvements over CNN:
- mAP: +0.108750
- F1 micro: +0.043144
- F1 macro: +0.053257
- precision micro: +0.043262
- recall micro: +0.040982
- Hamming loss: -0.020995 (lower is better)

## Per-class mean F1

| Class | CNN | CRNN |
|---|---:|---:|
| air_conditioner | 0.401631 | **0.443732** |
| children_playing | 0.575626 | **0.583054** |
| dog_bark | 0.528906 | **0.661252** |
| drilling | 0.517914 | **0.574483** |
| engine_idling | 0.548679 | **0.618921** |
| jackhammer | **0.701231** | 0.687707 |
| siren | 0.578931 | **0.632981** |
| car_horn | 0.663692 | **0.740540** |

CRNN is higher on 7 of 8 target classes. CNN retains a small advantage on jackhammer.

## Single vs controlled-mixture performance

CNN:
- single: F1 micro 0.550776 ± 0.005033, F1 macro 0.586631 ± 0.005033, mAP 0.695764 ± 0.010346
- mix: F1 micro 0.548797 ± 0.008190, F1 macro 0.551792 ± 0.007778, mAP 0.628511 ± 0.007605

CRNN:
- single: F1 micro 0.614831 ± 0.018447, F1 macro 0.664000 ± 0.023633, mAP 0.822535 ± 0.011912
- mix: F1 micro 0.576376 ± 0.004244, F1 macro 0.582816 ± 0.008748, mAP 0.691822 ± 0.003890

Both models lose ranking performance on controlled mixtures versus single-event samples. CRNN remains stronger on both subsets.

## Overlap analysis

Mean micro-F1:
- overlap 0.25: CNN 0.528015, CRNN 0.539953
- overlap 0.50: CNN 0.552238, CRNN 0.583650
- overlap 1.00: CNN 0.565330, CRNN 0.605508

In this controlled mixture construction, larger overlap was not harder on average. Treat this as an empirical property of this protocol, not a universal claim.

## Relative-level analysis

Mean micro-F1:
- -6 dB: CNN 0.553602, CRNN 0.567973
- 0 dB: CNN 0.559098, CRNN 0.588966
- +6 dB: CNN 0.533626, CRNN 0.571980

Performance is strongest around equal source level for both models.

## OOD rejection — major limitation

Rule:
`No known class crosses its validation-selected threshold -> reject`

| Metric | CNN | CRNN |
|---|---:|---:|
| Known false rejection | 0.010231 ± 0.002864 | **0.004533 ± 0.001471** |
| Known acceptance | 0.989769 ± 0.002864 | **0.995467 ± 0.001471** |
| OOD rejection | 0.012674 ± 0.007915 | **0.046895 ± 0.025317** |
| OOD false acceptance | 0.987326 ± 0.007915 | **0.953105 ± 0.025317** |

The threshold-only rejection rule is not an effective OOD detector on the two held-out UrbanSound8K classes. Most OOD samples are still accepted as one or more known classes. This negative result must be reported explicitly and must not be repaired by tuning on held-out OOD data.

The demo may still expose the transparent `No confident known class` rule, but it must be described as a confidence heuristic, not robust open-set recognition.

## Final scientific interpretation

Supported:
- CRNN is the stronger architecture for the frozen window-level multi-label tagging task.
- The CRNN advantage appears across ranking metrics, thresholded F1 metrics, and most classes.
- Controlled mixtures are harder than single-event samples in ranking performance.
- Mixture overlap and relative level affect performance.
- The simple rejection heuristic is insufficient for reliable OOD detection.

Not supported:
- exact onset/offset SED
- source separation
- source counting
- localization
- calibrated probabilities
- robust open-set recognition

## Frozen-results rule

After observing held-out results:
- no threshold retuning
- no architecture change based on held-out performance
- no preprocessing change based on held-out performance
- no target-class or split change
- no metric-definition change
- no best-seed substitution

Any future development must be treated as a new experimental protocol.
