.PHONY: setup data train test evaluate hdfs-lstm serve

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

hdfs-lstm:
	.venv/bin/python scripts/train_hdfs_lstm.py --per-class 5000 --epochs 15

serve:
	.venv/bin/python -m uvicorn app.api:app --reload
