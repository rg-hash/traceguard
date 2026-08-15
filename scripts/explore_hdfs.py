from itertools import islice
from pathlib import Path
import csv
import re

LOG_PATH = Path("data/raw/HDFS_v1/HDFS.log")
LABEL_PATH = Path("data/raw/HDFS_v1/preprocessed/anomaly_label.csv")

# Read labels into a dictionary:
# {"blk_123": "Normal", "blk_456": "Anomaly"}
with LABEL_PATH.open() as file:
    labels = {
        row["BlockId"]: row["Label"]
        for row in csv.DictReader(file)
    }

block_pattern = re.compile(r"blk_-?\d+")

with LOG_PATH.open(encoding="utf-8", errors="replace") as file:
    for line in file:  
        block_ids = block_pattern.findall(line)

        for block_id in block_ids:
            print({
                "block_id": block_id,
                "label": labels.get(block_id, "Unknown"),
                "log": line.strip()
            })