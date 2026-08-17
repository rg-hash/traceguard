from pathlib import Path
import json
import random

LOG_PATH = Path("data/raw/BGL/BGL.log")

DEVELOPMENT_OUTPUT_PATH = Path(
    "data/processed/bgl_development_windows.jsonl"
)
LOCKED_OUTPUT_PATH = Path(
    "data/processed/bgl_locked_temporal_holdout.jsonl"
)

WINDOW_SIZE = 20
DEVELOPMENT_SAMPLES_PER_CLASS = 5000
LOCKED_SAMPLES_PER_CLASS = 1000

DEVELOPMENT_SEED = 7
LOCKED_SEED = 17


def create_event(line: str) -> dict:
    parts = line.strip().split(maxsplit=2)

    # Exclude the BGL alert label from the message to prevent label leakage.
    message = parts[2] if len(parts) >= 3 else line.strip()

    return {
        "timestamp": " ".join(parts[:2]) if len(parts) >= 2 else "unknown",
        "service": "bluegene-l",
        "severity": "ALERT" if parts[0] != "-" else "INFO",
        "message": message,
    }


def count_complete_windows() -> int:
    nonempty_lines = 0

    with LOG_PATH.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            if line.strip():
                nonempty_lines += 1

    return nonempty_lines // WINDOW_SIZE


def reservoir_add(
    samples: dict[int, list[dict]],
    seen: dict[int, int],
    label: int,
    incident: dict,
    limit: int,
    rng: random.Random,
) -> None:
    """Uniformly sample at most `limit` incidents without storing all windows."""
    seen[label] += 1

    if len(samples[label]) < limit:
        samples[label].append(incident)
        return

    replacement_index = rng.randrange(seen[label])

    if replacement_index < limit:
        samples[label][replacement_index] = incident


def write_jsonl(path: Path, samples: dict[int, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output:
        for label in [0, 1]:
            for incident in samples[label]:
                output.write(json.dumps(incident) + "\n")


def main() -> None:
    total_windows = count_complete_windows()
    development_window_limit = int(total_windows * 0.80)

    print(f"Total complete BGL windows: {total_windows:,}")
    print(
        "Development timeline: "
        f"windows [0, {development_window_limit - 1:,}]"
    )
    print(
        "Locked future timeline: "
        f"windows [{development_window_limit:,}, {total_windows - 1:,}]"
    )

    development_samples = {0: [], 1: []}
    locked_samples = {0: [], 1: []}

    development_seen = {0: 0, 1: 0}
    locked_seen = {0: 0, 1: 0}

    development_rng = random.Random(DEVELOPMENT_SEED)
    locked_rng = random.Random(LOCKED_SEED)

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

            if len(window) == WINDOW_SIZE:
                label = 1 if window_has_alert else 0

                incident = {
                    "incident_id": f"bgl-window-{window_number:06d}",
                    "is_anomaly": label,
                    "root_cause": "unknown",
                    "source": "LogHub BGL",
                    "events": window,
                }

                if window_number < development_window_limit:
                    reservoir_add(
                        samples=development_samples,
                        seen=development_seen,
                        label=label,
                        incident=incident,
                        limit=DEVELOPMENT_SAMPLES_PER_CLASS,
                        rng=development_rng,
                    )
                else:
                    reservoir_add(
                        samples=locked_samples,
                        seen=locked_seen,
                        label=label,
                        incident=incident,
                        limit=LOCKED_SAMPLES_PER_CLASS,
                        rng=locked_rng,
                    )

                window = []
                window_has_alert = False
                window_number += 1

            if line_number % 1_000_000 == 0:
                print(f"Processed {line_number:,} BGL log lines...")

    write_jsonl(DEVELOPMENT_OUTPUT_PATH, development_samples)
    write_jsonl(LOCKED_OUTPUT_PATH, locked_samples)

    print("\nDevelopment windows available:")
    print(f"  Normal: {development_seen[0]:,}")
    print(f"  Anomaly: {development_seen[1]:,}")
    print("Development windows saved:")
    print(f"  Normal: {len(development_samples[0]):,}")
    print(f"  Anomaly: {len(development_samples[1]):,}")

    print("\nLocked future windows available:")
    print(f"  Normal: {locked_seen[0]:,}")
    print(f"  Anomaly: {locked_seen[1]:,}")
    print("Locked future windows saved:")
    print(f"  Normal: {len(locked_samples[0]):,}")
    print(f"  Anomaly: {len(locked_samples[1]):,}")

    print(f"\nCreated development data: {DEVELOPMENT_OUTPUT_PATH}")
    print(f"Created locked holdout: {LOCKED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()