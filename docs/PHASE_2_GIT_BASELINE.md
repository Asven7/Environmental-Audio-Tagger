# Phase 2 — Repository Baseline and Local Git

## Goal

Create the first trustworthy local Git history from a project state that is already verified on the target Windows machine.

## Preconditions

- Python virtual environment is active when running project tests.
- `python -m pytest` passes all 13 tests locally.
- Git is installed.

## Changes prepared in this phase

- hardened `.gitignore`;
- added `.gitattributes`;
- removed the unused `.env.example`;
- added `docs/GIT_WORKFLOW.md`;
- documented the Git artifact and configuration decisions.

## Local phase gate

The phase is complete only after the user verifies:

1. `git init` succeeds;
2. branch is named `main`;
3. ignore rules protect `.venv`, `.pytest_tmp`, `.env`, and dataset paths;
4. the tiny synthetic demo checkpoint remains trackable;
5. staged files contain no private/large local data;
6. the baseline commit succeeds;
7. `git status` is clean;
8. `git log --oneline --decorate -5` shows the baseline commit.

GitHub is intentionally not configured in this phase.
