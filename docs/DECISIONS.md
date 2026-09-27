# Architectural and Experimental Decisions

## ADR-001 — Task Definition

**Decision:** implement window-level multi-label audio tagging / event-presence detection, not full SED.

**Rationale:** the proposal output is one multi-label vector per short window; exact onset/offset estimation was already outside scope.

**Consequence:** repository terminology avoids implying exact temporal event boundaries.

## ADR-002 — UrbanSound8K as Main Dataset

**Decision:** keep UrbanSound8K as the primary undergraduate dataset and construct controlled mixtures after split assignment.

**Rationale:** manageable size, official folds, simple local reproduction.

**Limitation:** it is fundamentally single-label; synthetic mixtures are controlled experiments, not proof of real soundscape generalization.

## ADR-003 — Fixed Split Protocol

**Decision:** folds 1–7 train, 8 validation, 9–10 test by default.

**Rationale:** simple, reproducible, impossible to reinterpret after test inspection if frozen before experiments.

## ADR-004 — Eight Known + Two Held-Out Classes

**Decision:** default known classes are eight UrbanSound8K classes; `gun_shot` and `street_music` are held out.

**Rationale:** keeps the main classifier limited while making a separate held-out-class rejection experiment possible.

**Important:** held-out classes are not a proxy for all unknown sounds.

## ADR-005 — Controlled Relative-Level and Overlap Factors

**Decision:** generate two-source mixture manifests with explicit dB differences and overlap ratios.

**Rationale:** distinguishes relative-level difficulty from temporal-overlap difficulty and makes both measurable.

## ADR-006 — Log-Mel as the Main Representation

**Decision:** do not add MFCC/RMS/ZCR to the primary model.

**Rationale:** Log-Mel preserves a direct time-frequency structure useful to CNN/CRNN and avoids unnecessary feature branching.

**Future work:** classical features can be added as an auxiliary baseline if there is a specific research question.

## ADR-007 — GRU as Default Recurrent Unit

**Decision:** default CRNN uses one unidirectional GRU.

**Rationale:** fewer parameters and simpler compute than LSTM while still modeling within-window temporal dependencies.

**Implementation:** LSTM remains supported through configuration but is not part of the default experiment matrix.

## ADR-008 — Validation mAP for Model Selection

**Decision:** choose best checkpoint by validation mAP, then tune thresholds using per-class validation F1.

**Rationale:** mAP is threshold-independent for model selection; threshold tuning is then explicit and validation-only.

## ADR-009 — Rejection Terminology

**Decision:** return `no_confident_known_class` when no class crosses its threshold.

**Rationale:** avoids the academically incorrect implication that a simple sigmoid threshold solves open-set recognition.

## ADR-010 — No Database

**Decision:** use CSV/JSON experiment artifacts instead of SQL/NoSQL.

**Rationale:** the project stores immutable manifests, model artifacts, and metrics rather than transactional application data. A database would add no meaningful value.

## ADR-011 — No REST API / Authentication

**Decision:** local in-process inference and localhost UI only.

**Rationale:** an API/user system does not contribute to the project’s signal-processing/ML research questions.

## ADR-012 — No Mandatory Docker

**Decision:** use Python virtual environments as the primary setup path.

**Rationale:** microphone and optional GPU device access are more cumbersome in containers; dependency files already provide adequate reproduction for this scope.

## ADR-013 — Synthetic Demo Artifacts

**Decision:** include a generated synthetic tone/noise dataset and tiny checkpoints.

**Rationale:** permits immediate software verification without redistributing UrbanSound8K.

**Rule:** demo metrics must never be presented as environmental-sound research results.

## ADR-014 — Git Artifact Policy

**Decision:** keep source, tests, configuration, documentation, generated synthetic sample data, and the two tiny synthetic demo model artifacts in Git; ignore real datasets, normal training checkpoints, and generated experiment outputs.

**Context:** a professor should be able to clone the repository and inspect or smoke-test it, but UrbanSound8K and real experiment artifacts should not bloat version control.

**Alternatives:** commit every artifact; ignore all model files including the tiny demo checkpoints.

**Reason:** the selected policy preserves a lightweight, reproducible demonstration while keeping research data and normal model outputs outside Git.

**Trade-off:** binary demo checkpoints have limited diffability, but their combined size is tiny and they are explicitly non-scientific smoke artifacts.

## ADR-015 — No `.env` Runtime Configuration

**Decision:** remove the unused `.env.example` and use YAML configuration plus explicit CLI arguments for current runtime settings.

**Context:** the project requires no API keys, passwords, database URLs, or other secrets, and the existing application did not read the previously documented `ESAUDIO_*` variables.

**Alternatives:** add a dotenv dependency and environment-variable loading solely to preserve `.env.example`.

**Reason:** adding configuration machinery that is not needed would increase complexity and create misleading documentation.

**Trade-off:** a future external service may justify introducing `.env` support; `.env` files remain ignored by Git for that case.
