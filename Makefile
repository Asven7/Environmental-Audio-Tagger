.PHONY: install test demo ui clean

install:
	python -m pip install -e '.[ui,dev]'

test:
	pytest

demo:
	python scripts/run_demo_pipeline.py

ui:
	python scripts/run_ui.py --checkpoint artifacts/demo_crnn/best_model.pt --thresholds artifacts/demo_crnn/thresholds.json

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .pytest_cache
