# Documentation Index

This directory contains the project protocol, implementation history, frozen research results, engineering verification, and defense material.

For a first read, use this order:

1. [`../README.md`](../README.md) — project overview, final results, installation, and limitations.
2. [`SCOPE_AND_ACCEPTANCE.md`](SCOPE_AND_ACCEPTANCE.md) — exact scientific scope and exclusions.
3. [`ARCHITECTURE.md`](ARCHITECTURE.md) — model and data-flow architecture.
4. [`CLEAN_INSTALL.md`](CLEAN_INSTALL.md) — verified fresh-clone installation.
5. [`PHASE_11_FINAL_RESULTS.md`](PHASE_11_FINAL_RESULTS.md) — frozen three-seed CNN/CRNN results.
6. [`PHASE_12_RUNTIME_RESULTS.md`](PHASE_12_RUNTIME_RESULTS.md) — canonical runtime measurements.
7. [`PHASE_13_END_TO_END_DEMO.md`](PHASE_13_END_TO_END_DEMO.md) — file/UI/microphone acceptance.
8. [`PHASE_14_QA.md`](PHASE_14_QA.md) — repository QA gate.
9. [`PHASE_15_GITHUB_CI.md`](PHASE_15_GITHUB_CI.md) — hosted CI boundary.
10. [`PHASE_16_COMPLETION.md`](PHASE_16_COMPLETION.md) — clean-install and documentation closure.

## Scientific protocol

- [`DATASET.md`](DATASET.md)
- [`DECISIONS.md`](DECISIONS.md)
- [`EVALUATION.md`](EVALUATION.md)
- [`REQUIREMENTS_TRACEABILITY.md`](REQUIREMENTS_TRACEABILITY.md)
- [`VERIFICATION.md`](VERIFICATION.md)

## Reproducibility and operations

- [`SETUP_LOCAL.md`](SETUP_LOCAL.md)
- [`CLEAN_INSTALL.md`](CLEAN_INSTALL.md)
- [`TESTING.md`](TESTING.md)
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
- [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md)
- [`SECURITY_PRIVACY.md`](SECURITY_PRIVACY.md)

## Demo and defense

- [`DEFENSE_DEMO.md`](DEFENSE_DEMO.md)
- [`IMPLEMENTATION_REPORT_FA.md`](IMPLEMENTATION_REPORT_FA.md)

## Phase records

The `PHASE_*` files preserve the incremental implementation and verification record. They are useful for auditability, but the top-level README and the primary documents above should be preferred for normal project use.

## Important scientific boundary

The frozen held-out test result is not a development set.

Do not retune thresholds, alter preprocessing, select another model/seed, or change the protocol using the frozen held-out metrics. Ordinary installation, CI, documentation, and demo work must not rerun the frozen held-out evaluation.
