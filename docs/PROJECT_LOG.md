# Project Log

This log keeps the project history. Early entries describe the state **at that time**; later entries record the completed real-data, deployment, QA, CI, and clean-install work.

## 2026-09-26 — Phase 0: Existing Implementation Audit

### Work completed
- reviewed the proposal-derived task boundaries;
- audited the previously generated repository structure and core modules;
- re-ran syntax compilation;
- confirmed the expected `src/`-layout import failure before package installation;
- installed the package in editable mode in the agent environment;
- re-ran the then-current automated suite (`13 passed`);
- re-ran the synthetic CNN/CRNN engineering smoke pipeline;
- locked the phase-gated roadmap.

### Technical decisions
- retain a single local Python research application;
- keep database/API/authentication/microservices out of scope;
- keep GRU as the default recurrent block;
- treat synthetic demo outputs as software verification, not research metrics.

## 2026-09-27 — Phase 1: Local Environment and Windows pytest fix

User environment:

```text
Windows 10 Enterprise 64-bit
Python 3.11.9
i7-12700H
16 GB RAM
RTX 3050 Ti Laptop GPU / 4 GB VRAM
NVIDIA driver 566.07
```

Verified:
- native project `.venv`;
- PyTorch/torchaudio CUDA path;
- CUDA tensor smoke operation;
- `python -m pip check`;
- repository-local pytest temp workaround.

The Windows user-temp ACL failure was resolved by configuring:

```text
--basetemp=.pytest_tmp
```

and the normal repository command subsequently passed all 13 tests.

## Phase 2 — Repository baseline / Git

Completed:
- `.gitignore` hardening;
- `.gitattributes`;
- no-secrets/YAML+CLI configuration decision;
- Git artifact policy;
- documented local Git workflow;
- stable baseline commit.

## Phases 3–6 — Audio, dataset, mixtures, Log-Mel

Completed and user verified:
- config/audio foundations;
- UrbanSound8K leakage-safe protocol;
- broad `fsID` crossing audit/exclusion;
- controlled same-split two-source mixtures;
- Log-Mel feature pipeline and CPU/CUDA feature parity.

Real protocol retained:

```text
train folds 1–7
validation fold 8
test folds 9–10
8 known target classes
2 held-out classes
```

## Phases 7–10 — Models, training, thresholds, frozen evaluation protocol

Completed and user verified:
- lightweight CNN baseline;
- CRNN with unidirectional GRU;
- reproducible training/checkpointing;
- validation-mAP model selection;
- validation-only per-class thresholds;
- frozen known/OOD held-out evaluation workflow.

No threshold or model selection uses held-out test results.

## Phase 11 — Official multi-seed scientific experiment

Official matrix completed:

```text
CNN  × 13,23,37
CRNN × 13,23,37
```

Frozen held-out result headline:

```text
CNN  mAP  0.618628 ± 0.008281
CRNN mAP  0.727378 ± 0.007039
```

CRNN was stronger on the primary frozen known-class metrics.

The threshold-only held-out rejection rule was weak; CRNN OOD rejection was only about 4.69%. This limitation was retained rather than tuned away.

## Phase 12 — Frozen deployment and runtime

Deployment selected from validation only:

```text
CRNN seed 23
validation mAP = 0.6757137110147023
```

Canonical p95:

```text
CPU  4.358 ms
CUDA 1.561 ms
hop  1000 ms
```

Both paths passed the no-backlog criterion.

## Phase 13 — File/UI/microphone acceptance

User verified:
- sequential file windows;
- Gradio file UI;
- browser microphone;
- live history ordering;
- Stop preserving results/history;
- explicit Clear results;
- `sounddevice` device listing;
- physical microphone capture;
- CPU live inference.

Live background/fan false positives were documented as a model/OOD limitation. Thresholds were not changed.

## Phase 14 — Repository QA

Verified:
- dedicated QA tests;
- full pytest;
- `git diff --check`;
- `pip check`;
- compilation;
- UI/mic CLI surfaces;
- Phase-13 acceptance;
- frozen deployment integrity;
- tracked-file hygiene.

The QA gate reported all seven checks PASS.

## Phase 15 — GitHub / CI

Completed:
- public GitHub repository;
- GitHub Actions CPU/Python-3.11 CI;
- local CI contract verification;
- push to `main`;
- green hosted CI.

Hosted CI intentionally does not require the dataset, frozen research artifacts, microphone hardware, or CUDA.

## Phase 16A — Clean installation

A separate fresh clone and fresh `.venv` were verified with Python 3.11.9 and CPU PyTorch.

Final acceptance:

```text
pip check: PASS
clean-install verifier: PASS
full pytest: 173 passed
git working tree: clean
```

No dataset/frozen experiment artifact was required and held-out metrics were not recomputed.

## Phase 16B — Final README/documentation consolidation

Verified:

```text
documentation contract: 8 passed
full pytest: 181 passed
git diff --check: PASS
commit: 539418a
push: main -> origin/main
GitHub Actions: GREEN
```

## Phase 16C — Final consistency audit

A repository-wide audit identified stale historical/current-status contradictions in status, traceability, setup, testing, verification, project-log, changelog, phase-status, Makefile, and reference dependency documents.

This cleanup is documentation/metadata only. It must not alter:
- model code,
- config scientific protocol,
- checkpoints,
- thresholds,
- held-out results,
- runtime result artifacts,
- deployment selection.

## Next step

After Phase 16C local verification, commit/push, and green CI:

```text
Phase 17 — University Demo and Defense Preparation
```
