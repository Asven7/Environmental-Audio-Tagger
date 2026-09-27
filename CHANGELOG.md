# Changelog

## Unreleased

### Phase 0 audit baseline
- audited the previously generated implementation instead of treating it as final;
- re-verified automated tests after editable installation;
- re-verified the synthetic end-to-end CNN/CRNN smoke pipeline;
- documented required corrections and a phase-gated implementation roadmap;
- explicitly kept real UrbanSound8K metrics and physical microphone verification pending.

### Phase 1 local environment preparation
- added a dependency-free local environment diagnostic script;
- added Windows-focused local setup instructions;
- documented the preferred PyTorch 2.10.0 CUDA 12.6 installation and CPU fallback;
- added troubleshooting guidance for Python path, virtual environments, CUDA detection, and 4 GB VRAM limits.


### Fixed
- Route pytest temporary files to a repository-local `.pytest_tmp/` directory to avoid Windows user-temp ACL failures observed during local verification.

### Documentation
- Record the verified Windows/PyTorch/CUDA environment and pytest temp-directory troubleshooting workflow.

### Phase 2 repository baseline
- add a documented local Git workflow;
- add `.gitattributes` to normalize text line endings across Windows/Linux;
- harden `.gitignore` for virtual environments, datasets, temporary files, secrets, and generated artifacts;
- keep only the deliberately small synthetic demo artifacts trackable;
- remove the misleading unused `.env.example` and document the no-secrets/YAML+CLI configuration decision.
