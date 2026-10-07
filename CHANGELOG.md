# Changelog

## Unreleased

### Phase 16C — final repository consistency cleanup
- reconcile stale status documents with the already verified Phase-11 through Phase-16 results;
- update setup, testing, troubleshooting, verification, traceability, project-log, and Persian implementation-report documents;
- remove the misleading stale pseudo-lock file, keep `pyproject.toml` as the dependency contract, and refresh Makefile convenience commands;
- preserve the frozen scientific protocol, checkpoint selection, thresholds, and held-out results;
- keep Phase 17 defense/demo preparation separate from this documentation cleanup.

### Phase 16 — documentation and clean-install reproducibility
- add a fresh-clone / fresh-virtual-environment clean-install verifier;
- verify non-editable installation with `.[all]` on Python 3.11.9;
- verify `pip check`, console entry points, optional UI/live imports, CNN/CRNN synthetic CPU forwards, and the complete repository test suite;
- complete the final top-level README and documentation index;
- verify Phase-16B with 8 documentation-contract tests and a full `181 passed` regression run;
- push the final Phase-16B commit and verify GitHub Actions green.

### Phase 15 — GitHub and CI
- add `.github/workflows/ci.yml`;
- run CPU/Python-3.11 repository checks on pushes, pull requests, and manual dispatch;
- verify CI locally and on GitHub without requiring UrbanSound8K, frozen research artifacts, CUDA, microphone hardware, or repository secrets.

### Phase 14 — repository QA
- add a repository-level QA gate covering `git diff --check`, `pip check`, compilation, CLI surfaces, Phase-13 acceptance, full pytest, frozen deployment integrity, and tracked-file hygiene;
- verify all seven QA checks locally.

### Phase 13 — file UI and microphone demo
- add sequential file-window analysis and rolling microphone inference;
- add Gradio file/browser-microphone UI and local `sounddevice` CLI;
- verify browser microphone, local microphone capture, history ordering, stop preservation, clear-results behavior, and CPU end-to-end inference;
- retain the observed live false-positive / weak-OOD limitation instead of retuning after held-out evaluation.

### Phase 12 — frozen inference and runtime
- select CRNN seed 23 using validation mAP only;
- verify frozen checkpoint/threshold integrity;
- measure canonical batch-1 runtime on CPU and CUDA;
- verify p95 total compute of 4.358 ms on CPU and 1.561 ms on CUDA, both below the 1000 ms stream hop.

### Phase 11 — multi-seed scientific experiments
- execute CNN and CRNN for seeds 13, 23, and 37 under the frozen protocol;
- freeze six model/threshold pairs before held-out aggregation;
- report final three-seed held-out results;
- identify CRNN as the stronger model family for the frozen known-class task;
- record weak held-out rejection as a major limitation.

### Phases 3–10 — core scientific pipeline
- harden configuration and waveform preprocessing;
- freeze the leakage-safe UrbanSound8K protocol and controlled mixture generation;
- implement and verify Log-Mel features, CNN and CRNN architectures;
- implement reproducible training, validation-only checkpoint selection, validation-only per-class threshold tuning, and frozen held-out evaluation.

### Phase 2 — repository baseline
- add the documented local Git workflow;
- add `.gitattributes` for cross-platform line-ending consistency;
- harden `.gitignore` for virtual environments, datasets, temporary files, secrets, and generated artifacts;
- keep only the deliberately small synthetic demo artifacts trackable;
- remove the misleading unused `.env.example`.

### Phase 1 — local environment preparation
- add a dependency-free local environment diagnostic script;
- document Windows setup and the preferred PyTorch CUDA/CPU installation paths;
- configure pytest to use repository-local `.pytest_tmp/` after a Windows temporary-directory ACL failure;
- verify the local Python 3.11 / RTX 3050 Ti development environment.

### Phase 0 — implementation audit baseline
- audit the proposal-derived implementation instead of treating the initial generated repository as final;
- lock the phase-gated implementation workflow and scientific scope;
- establish the synthetic engineering demo as software verification only, not research evidence.
