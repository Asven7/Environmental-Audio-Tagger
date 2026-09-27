# Git Workflow

## Purpose

Git is used from the verified local baseline onward to make project changes traceable, reversible, and defensible during the undergraduate project. The workflow is intentionally simple because this is primarily a single-developer academic repository.

## Repository policy

- `main` should remain runnable and should pass the relevant automated tests at milestone commits.
- Commit source code, tests, configuration, documentation, and small synthetic demo assets.
- Do **not** commit `.venv`, real UrbanSound8K audio, private `.env` files, temporary test data, large experiment outputs, or normal training checkpoints.
- The two tiny synthetic demo checkpoints under `artifacts/demo_cnn/` and `artifacts/demo_crnn/` are intentional exceptions. They exist only to support a reproducible smoke/demo flow and are not scientific results.
- Never commit credentials, passwords, tokens, or API keys.

## First-time initialization

Run from the repository root after the local environment and tests are verified:

```powershell
git init
git branch -M main
git status
```

Check your local Git identity:

```powershell
git config user.name
git config user.email
```

If either value is empty, set it for this repository only:

```powershell
git config --local user.name "YOUR NAME"
git config --local user.email "YOUR EMAIL"
```

Replace the placeholders with the identity you want recorded in commits. A GitHub no-reply email can be used later if privacy is preferred.

## Verify ignore rules before the first commit

```powershell
git check-ignore -v .venv/pyvenv.cfg
git check-ignore -v .pytest_tmp
git check-ignore -v .env
git check-ignore -v data/UrbanSound8K/example.wav
```

Each command should print the matching `.gitignore` rule. If a command prints nothing for an item that should stay local, stop and fix the ignore rules before staging.

The small synthetic demo model is intentionally trackable. Verify that Git considers it trackable:

```powershell
git check-ignore -q artifacts/demo_crnn/best_model.pt
if ($LASTEXITCODE -eq 1) { Write-Host "TRACKABLE: demo checkpoint is not ignored" } else { Write-Host "ERROR: demo checkpoint is ignored" }
```

Expected output:

```text
TRACKABLE: demo checkpoint is not ignored
```

## Stage and review the baseline

```powershell
git add .
git status --short
git diff --cached --stat
git diff --cached
```

Review the staged files before committing. In particular, confirm that `.venv/`, `.pytest_tmp/`, real datasets, and private files are absent.

## First milestone commit

```powershell
git commit -m "chore: establish verified project baseline"
```

Then inspect the history:

```powershell
git log --oneline --decorate -5
```

## Daily workflow

Before work:

```powershell
git status
git log --oneline --decorate -5
```

After a small coherent change:

```powershell
python -m pytest
git status
git diff
git add <files>
git diff --cached
git commit -m "<type>: <short description>"
```

Prefer focused commits. Do not use `git add .` automatically after every change; use explicit paths when only a few files were modified.

## Commit message convention

Use a lightweight Conventional Commits style:

- `feat:` new project behavior
- `fix:` bug fix
- `test:` test-only changes
- `docs:` documentation-only changes
- `refactor:` internal restructure without behavior change
- `chore:` repository/tooling/environment maintenance

Examples:

```text
feat: add controlled two-source mixture generation
fix: preserve relative event level after normalization
test: add leakage checks for manifest generation
docs: document UrbanSound8K preparation workflow
```

## Branches

For the current solo project, do not create branches for every tiny edit. Keep `main` stable and use a short-lived branch only for a change that may take several iterations or temporarily break the project.

Example:

```powershell
git switch -c feat/urbansound-manifests
```

After the work is verified and committed:

```powershell
git switch main
git merge --no-ff feat/urbansound-manifests
git branch -d feat/urbansound-manifests
```

GitHub pull requests will be introduced later only where they add review/CI value.

## Undo and inspection safety

Useful non-destructive commands:

```powershell
git status
git diff
git diff --cached
git log --oneline --decorate --graph --all
git show <commit>
```

To unstage a file without deleting its changes:

```powershell
git restore --staged <file>
```

To discard an unstaged file change, only after confirming that the change is not needed:

```powershell
git restore <file>
```

Do not use `git reset --hard`, `git clean -fd`, or force-push as routine fixes. They can destroy local work.

## GitHub boundary

This document covers local Git only. GitHub remote creation, visibility, push, CI, tags, and releases are handled in a later phase after the local repository has stable history.
