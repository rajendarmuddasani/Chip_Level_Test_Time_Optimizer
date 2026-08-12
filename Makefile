.PHONY: setup lint test evidence api dashboard compose clean

setup:
	python -m pip install --requirement requirements-test.txt

lint:
	python -m ruff check benchmark deployment/api.py deployment/runtime.py deployment/generate_flags.py evaluation/metrics.py scripts/run_public_benchmark.py scripts/run_operational_envelope_confirmation.py scripts/run_api_load.py scripts/scan_public_payload.py scripts/validate_evidence.py app.py tests/test_api.py tests/test_dashboard.py tests/test_documentation.py tests/test_drift.py tests/test_ensemble.py tests/test_evaluator.py tests/test_evidence.py tests/test_experiment.py tests/test_flag_generation.py tests/test_notebook.py tests/test_policy_metrics.py tests/test_privacy.py tests/test_runtime.py tests/test_synthetic_data.py tests/test_vae.py models/anomaly_detection/vae_model.py models/ensemble.py models/statistical/sigma_rules.py --select E4,E7,E9,F --ignore E402

test:
	python -m pytest tests -q --cov=benchmark --cov=deployment --cov=models --cov=evaluation --cov=preprocessing --cov-report=term --cov-fail-under=60

evidence:
	python scripts/validate_evidence.py

api:
	python -m uvicorn deployment.api:app --host 127.0.0.1 --port 8005

dashboard:
	python -m streamlit run app.py --server.port 8505

compose:
	docker compose up --build

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; find . -name '*.pyc' -delete 2>/dev/null; true
