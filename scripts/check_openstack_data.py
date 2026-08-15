import json
from collections import Counter

rows = []

with open("data/processed/openstack_windows.jsonl") as file:
    for line in file:
        if line.strip():
            rows.append(json.loads(line))

print("Total windows:", len(rows))

print(
    "Class distribution:",
    Counter(row["is_anomaly"] for row in rows)
)

print(
    "Unique VM instances:",
    len(set(row["source_instance_id"] for row in rows))
)

print(
    "Normal VMs:",
    len({
        row["source_instance_id"]
        for row in rows
        if row["is_anomaly"] == 0
    })
)

print(
    "Anomaly VMs:",
    len({
        row["source_instance_id"]
        for row in rows
        if row["is_anomaly"] == 1
    })
)