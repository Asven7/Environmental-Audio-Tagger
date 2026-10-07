.PHONY: install install-dev test verify-clean demo ui live-help clean

install:
	python -m pip install ".[all]"

install-dev:
	python -m pip install -e ".[all]"

test:
	python -m pytest

verify-clean:
	python scripts/verify_clean_install.py --project-root . --output artifacts/phase16_clean_install_report.json

demo:
	python scripts/run_demo_pipeline.py

ui:
	python scripts/run_ui.py --checkpoint artifacts/demo_crnn/best_model.pt --thresholds artifacts/demo_crnn/thresholds.json --device cpu

live-help:
	python scripts/live_microphone.py --help

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .pytest_tmp
