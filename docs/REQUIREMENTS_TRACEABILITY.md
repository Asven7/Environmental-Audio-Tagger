# Proposal-to-Implementation Traceability

| Requirement | Implementation | Verification/Test | Status |
|---|---|---|---|
| File audio input | `audio.py`, `inference.py` | checkpoint inference test + demo CLI | IMPLEMENTED AND VERIFIED |
| Microphone input | `scripts/live_microphone.py`, Gradio recording | stream buffer tests; physical device not available in build environment | IMPLEMENTED, HARDWARE VERIFICATION REQUIRED |
| 2 s window / 1 s hop default | `config/default.yaml`, `streaming.py` | config + stream-buffer tests | IMPLEMENTED AND VERIFIED |
| Mono conversion/resampling | `audio.py::load_audio` | exercised by pipeline; file loader path verified | IMPLEMENTED AND VERIFIED |
| STFT/Log-Mel | `features.py::LogMelExtractor` | feature shape test | IMPLEMENTED AND VERIFIED |
| CNN baseline | `models.py::CNNTagger` | forward test + synthetic training/evaluation | IMPLEMENTED AND VERIFIED |
| CRNN with recurrent temporal modeling | `models.py::CRNNTagger` | forward test + synthetic training/evaluation | IMPLEMENTED AND VERIFIED |
| GRU/LSTM option | `models.py` | GRU default verified; LSTM code path structurally implemented | GRU VERIFIED; LSTM OPTIONAL |
| Multi-label logits + BCEWithLogitsLoss | `training.py` | synthetic training smoke run | IMPLEMENTED AND VERIFIED |
| Sigmoid inference | `inference.py` | checkpoint round-trip/inference test | IMPLEMENTED AND VERIFIED |
| Per-class decision thresholds | `evaluation.py`, `checkpoints.py` | threshold fixture test + training smoke | IMPLEMENTED AND VERIFIED |
| Controlled two-label mixtures | `audio.py`, `manifests.py`, `dataset.py` | demo dataset test + synthetic runs | IMPLEMENTED AND VERIFIED |
| Split-before-mix leakage prevention | `manifests.py` | leakage assertion test | IMPLEMENTED AND VERIFIED |
| Relative-level analysis | mixture metadata + group evaluation | synthetic test evaluation output | IMPLEMENTED AND VERIFIED |
| Temporal-overlap analysis | mixture metadata + group evaluation | synthetic test evaluation output | IMPLEMENTED AND VERIFIED |
| Precision/Recall/F1 | `evaluation.py` | metric unit test | IMPLEMENTED AND VERIFIED |
| micro/macro F1 | `evaluation.py` | metric unit test | IMPLEMENTED AND VERIFIED |
| mAP | `evaluation.py` | synthetic evaluation | IMPLEMENTED AND VERIFIED |
| Hamming Loss | `evaluation.py` | metric unit test | IMPLEMENTED AND VERIFIED |
| Held-out rejection | `evaluation.py::rejection_metrics` | synthetic evaluation pipeline | IMPLEMENTED AND VERIFIED AS LIMITED METHOD |
| Unknown limitation | docs/UI wording | documentation review | IMPLEMENTED |
| Runtime measurement | `runtime.py`, benchmark script | synthetic CPU benchmark | IMPLEMENTED AND VERIFIED |
| Stable no-backlog criterion | runtime benchmark | demo p95 vs hop | IMPLEMENTED AND VERIFIED ON SYNTHETIC DEMO |
| File UI | `scripts/run_ui.py` | Gradio Blocks build check | IMPLEMENTED AND BUILD-VERIFIED |
| Microphone UI | Gradio microphone recording + live CLI | browser/device dependent | IMPLEMENTED, DEVICE VERIFICATION REQUIRED |
| Git-ready structure | repository layout + `.gitignore` | file inventory | IMPLEMENTED |
| Reproducible configs | YAML + seed handling | config tests + smoke pipeline | IMPLEMENTED AND VERIFIED |
| Multiple seeds | `run_experiments.py` | script implemented; full UrbanSound8K runs require dataset | IMPLEMENTED, REAL-DATA EXECUTION PENDING |
| Final real-data comparison | real UrbanSound8K required | no dataset available in build environment | NOT EXECUTED |
| Real multi-label field validation | external/manual data required | intentionally not fabricated | OPTIONAL / NOT IMPLEMENTED |
| Source separation/counting/localization | explicitly excluded | scope docs | OUT OF SCOPE |
| Exact onset/offset SED | explicitly excluded | task definition | OUT OF SCOPE |
