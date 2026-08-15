from collections import defaultdict
from pathlib import Path
import csv
import json
import random
import re

LOG_PATH = Path("data/raw/HDFS_v1/HDFS.log")
LABEL_PATH = Path("data/raw/HDFS_v1/preprocessed/anomaly_label.csv")
OUTPUT_PATH = Path("data/processed/hdfs_incidents.jsonl")

BLOCK_PATTERN = re.compile(r"blk_-?\d+")
SEED = 7
BLOCKS_PER_CLASS = 1000  # Start with 1,000 Normal + 1,000 Anomaly blocks


# 1. Read labels
with LABEL_PATH.open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

normal_blocks = [
    row["BlockId"]
    for row in rows
    if row["Label"] == "Normal"
]

anomaly_blocks = [
    row["BlockId"]
    for row in rows
    if row["Label"] == "Anomaly"
]


# 2. Select a balanced, reproducible subset
rng = random.Random(SEED)

selected_normal = rng.sample(normal_blocks, BLOCKS_PER_CLASS)
selected_anomaly = rng.sample(anomaly_blocks, BLOCKS_PER_CLASS)

selected_labels = {}

for block_id in selected_normal:
    selected_labels[block_id] = 0

for block_id in selected_anomaly:
    selected_labels[block_id] = 1


# 3. Stream the 1.5 GB log file, one line at a time
grouped_logs = defaultdict(list)

with LOG_PATH.open(encoding="utf-8", errors="replace") as file:
    for line_number, line in enumerate(file, start=1):
        block_ids = BLOCK_PATTERN.findall(line)

        for block_id in block_ids:
            if block_id in selected_labels:
                grouped_logs[block_id].append({
                    "message": line.strip()
                })

        if line_number % 1_000_000 == 0:
            print(f"Processed {line_number:,} log lines...")


# 4. Write selected blocks as TraceGuard incidents
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as output:
    for block_id, label in selected_labels.items():
        events = grouped_logs.get(block_id, [])

        # Ignore a selected block if no matching logs were found
        if not events:
            continue

        incident = {
            "incident_id": block_id,
            "is_anomaly": label,
            "root_cause": "unknown",
            "source": "LogHub HDFS_v1",
            "events": events
        }

        output.write(json.dumps(incident) + "\n")


print(f"\nCreated: {OUTPUT_PATH}")
print(f"Selected blocks: {len(selected_labels)}")
print(f"Blocks with logs: {len(grouped_logs)}")