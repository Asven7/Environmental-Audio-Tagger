# Project documentation

The documents below describe the current implementation and its recorded results. They are intended for understanding, running and checking the repository. A university report formatted to the institution's requirements is a separate deliverable.

| Document | Contents |
|---|---|
| [Architecture](ARCHITECTURE.md) | Python modules, data flow, CLI and local interface |
| [Dataset](DATASET.md) | UrbanSound8K classes, folds, source-isolation policy and mixture construction |
| [Methods](METHODS.md) | Preprocessing, CNN/CRNN, training and metric definitions |
| [Results](RESULTS.md) | Recorded frozen multi-seed outcomes, rejection and runtime |
| [Reproducibility](REPRODUCIBILITY.md) | Dependencies, installation, experiment artifacts, hashes and limitations |
| [Testing](TESTING.md) | Automated tests, hosted CI, local checks and verification scope |
| [Usage](USAGE.md) | File processing, Gradio and microphone commands |
| [Security and privacy](SECURITY_PRIVACY.md) | Local operation and handling of checkpoint/audio files |
| [References](REFERENCES.md) | Dataset, methods and primary libraries |

The repository includes example synthetic audio and small example model files so basic prediction can run without UrbanSound8K. Their metrics are not the real-data research results. Frozen held-out test metrics are for reporting, not for additional threshold or architecture selection.
