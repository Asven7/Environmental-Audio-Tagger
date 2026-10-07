# Proposal-to-Implementation Traceability

| Requirement | Implementation | Verification/Test | Final status |
|---|---|---|---|
| File audio input | `audio.py`, `inference.py`, `streaming.py` | Phase-13 file acceptance + automated acceptance | USER VERIFIED |
| Microphone input | `scripts/live_microphone.py`, Gradio browser recording | browser mic + physical sounddevice capture | USER VERIFIED |
| 2 s window / 1 s hop default | `config/default.yaml`, `streaming.py` | config, streaming tests, Phase-13 acceptance | USER VERIFIED |
| Mono conversion/resampling | `audio.py::load_audio` | audio I/O tests + file demo | USER VERIFIED |
| STFT/Log-Mel | `features.py::LogMelExtractor` | feature tests, CPU/CUDA smoke | USER VERIFIED |
| CNN baseline | `models.py::CNNTagger` | architecture tests + three-seed official training/evaluation | USER VERIFIED |
| CRNN temporal model | `models.py::CRNNTagger` | architecture tests + three-seed official training/evaluation | USER VERIFIED |
| GRU/LSTM option | `models.py` | GRU frozen protocol verified; LSTM path implemented | GRU VERIFIED; LSTM OPTIONAL |
| Multi-label logits + BCEWithLogitsLoss | `training.py` | training protocol tests + official runs | USER VERIFIED |
| Sigmoid inference | `inference.py` | checkpoint inference + Phase-13 acceptance | USER VERIFIED |
| Per-class decision thresholds | `evaluation.py`, `checkpoints.py` | validation-only threshold selection + freeze integrity | USER VERIFIED |
| Controlled two-label mixtures | `audio.py`, `manifests.py`, `dataset.py` | real manifest audit + mixture tests | USER VERIFIED |
| Split-before-mix leakage prevention | `manifests.py` | source-group leakage checks + real audit | USER VERIFIED |
| Relative-level analysis | mixture metadata + grouped evaluation | frozen three-seed grouped results | USER VERIFIED |
| Temporal-overlap analysis | mixture metadata + grouped evaluation | frozen three-seed grouped results | USER VERIFIED |
| Precision/Recall/F1 | `evaluation.py` | metric tests + frozen results | USER VERIFIED |
| micro/macro F1 | `evaluation.py` | metric tests + frozen results | USER VERIFIED |
| mAP | `evaluation.py` | frozen results | USER VERIFIED |
| Hamming Loss | `evaluation.py` | frozen results | USER VERIFIED |
| Held-out rejection | threshold-based rejection metrics | frozen OOD results | USER VERIFIED AS LIMITED METHOD |
| Unknown/open-set limitation | docs + UI wording | final result review + live behavior | USER VERIFIED LIMITATION |
| Multiple seeds | official Phase-11 runner | CNN/CRNN × seeds 13,23,37 | USER VERIFIED |
| Final real-data comparison | UrbanSound8K frozen protocol | `PHASE_11_FINAL_RESULTS.md` | USER VERIFIED |
| Frozen model selection | validation-only selector | CRNN seed 23, val mAP 0.6757137110 | USER VERIFIED |
| Runtime measurement | frozen runtime benchmark | CPU and CUDA 100-iteration canonical benchmark | USER VERIFIED |
| No-backlog criterion | p95 total compute < 1 s hop | CPU 4.358 ms; CUDA 1.561 ms | USER VERIFIED |
| File UI | `scripts/run_ui.py` | Phase-13 manual + automated acceptance | USER VERIFIED |
| Browser microphone UI | Gradio microphone streaming | Phase-13 manual acceptance | USER VERIFIED |
| Local microphone CLI | `scripts/live_microphone.py` | physical device capture | USER VERIFIED |
| Git-ready structure | Git policy + `.gitignore` + `.gitattributes` | QA tracked-file hygiene | USER VERIFIED |
| Reproducible configs | YAML + seeds + frozen hashes | protocol/experiment tests | USER VERIFIED |
| Repository QA | `scripts/run_phase14_qa.py` | seven QA checks | USER VERIFIED |
| GitHub CI | `.github/workflows/ci.yml` | push CI on `main` | CI VERIFIED |
| Clean installation | `scripts/verify_clean_install.py` | fresh clone/fresh `.venv`, `pip check`, full pytest | USER VERIFIED |
| Real multi-label field validation | external annotated recordings required | not fabricated by repository | OPTIONAL / FUTURE |
| Source separation/counting/localization | explicitly excluded | scope docs | OUT OF SCOPE |
| Exact onset/offset SED | explicitly excluded | task definition | OUT OF SCOPE |

## Frozen scientific boundary

The final held-out results are already observed and frozen. Future model, preprocessing, class, split, threshold, or metric-definition changes require a newly declared experimental protocol before another held-out evaluation.
