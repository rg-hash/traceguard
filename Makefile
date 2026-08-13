.PHONY: setup data train test evaluate serve

setup:
	python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

data:
	.venv/bin/python scripts/generate_demo_data.py

train:
	.venv/bin/python scripts/train.py

test:
	.venv/bin/python -m pytest -q

evaluate:
	.venv/bin/python scripts/evaluate.py

serve:
	.venv/bin/python -m uvicorn app.api:app --reload
