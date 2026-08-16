from pathlib import Path
import json
import random

LOG_PATH = Path("data/raw/BGL/BGL.log")
OUTPUT_PATH = Path("data/processed/bgl_windows.jsonl")

WINDOW_SIZE = 20
SAMPLES_PER_CLASS = 5000
SEED = 7


def create_event(line):
    parts = line.strip().split(maxsplit=2)

    # First field is the BGL alert label.
    # Do not include it in message: that would leak the answer to the model.
    message = parts[2] if len(parts) >= 3 else line.strip()

    return {
        "timestamp": " ".join(parts[:2]) if len(parts) >= 2 else "unknown",
        "service": "bluegene-l",
        "severity": "ALERT" if parts[0] != "-" else "INFO",
        "message": message
    }


rng = random.Random(SEED)

# Reservoir samples: keep at most 5,000 windows per class.
samples = {
    0: [],
    1: []
}

seen = {
    0: 0,
    1: 0
}

window = []
window_has_alert = False
window_number = 0

with LOG_PATH.open(encoding="utf-8", errors="replace") as file:
    for line_number, line in enumerate(file, start=1):
        parts = line.strip().split(maxsplit=1)

        if not parts:
            continue

        is_alert = parts[0] != "-"

        window.append(create_event(line))
        window_has_alert = window_has_alert or is_alert

        # Make one non-overlapping 20-event incident window
        if len(window) == WINDOW_SIZE:
            label = 1 if window_has_alert else 0

            incident = {
                "incident_id": f"bgl-window-{window_number:06d}",
                "is_anomaly": label,
                "root_cause": "unknown",
                "source": "LogHub BGL",
                "events": window
            }

            seen[label] += 1

            if len(samples[label]) < SAMPLES_PER_CLASS:
                samples[label].append(incident)
            else:
                replacement_index = rng.randrange(seen[label])

                if replacement_index < SAMPLES_PER_CLASS:
                    samples[label][replacement_index] = incident

            window = []
            window_has_alert = False
            window_number += 1

        if line_number % 1_000_000 == 0:
            print(f"Processed {line_number:,} BGL log lines...")


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as output:
    for label in [0, 1]:
        for incident in samples[label]:
            output.write(json.dumps(incident) + "\n")

print(f"\nNormal windows available: {seen[0]:,}")
print(f"Anomaly windows available: {seen[1]:,}")
print(f"Saved normal samples: {len(samples[0]):,}")
print(f"Saved anomaly samples: {len(samples[1]):,}")
print(f"Created: {OUTPUT_PATH}")