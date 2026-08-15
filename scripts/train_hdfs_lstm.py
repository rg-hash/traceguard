"""DeepLog-inspired, sequence-aware HDFS anomaly benchmark.

The CSV is streamed and reservoir-sampled: it is never loaded in full memory.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ARTIFACT_DIR, ROOT

TRACE_PATH = ROOT / "data" / "raw" / "HDFS_v1" / "preprocessed" / "Event_traces.csv"
EVENT_ID = re.compile(r"E(\d+)")


def reservoir_sample(per_class: int, seed: int) -> tuple[list[list[int]], list[int]]:
    """Uniformly sample Success and Fail traces one CSV row at a time."""
    rng = random.Random(seed)
    samples: dict[int, list[list[int]]] = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    with TRACE_PATH.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["Label"] not in {"Success", "Fail"}:
                continue
            label = 0 if row["Label"] == "Success" else 1
            sequence = [int(event) for event in EVENT_ID.findall(row["Features"])]
            if not sequence:
                continue
            seen[label] += 1
            if len(samples[label]) < per_class:
                samples[label].append(sequence)
            else:
                candidate = rng.randrange(seen[label])
                if candidate < per_class:
                    samples[label][candidate] = sequence
    if not samples[0] or not samples[1]:
        raise ValueError("No usable Success/Fail sequences found in Event_traces.csv")
    return samples[0] + samples[1], [0] * len(samples[0]) + [1] * len(samples[1])


def main(per_class: int, max_length: int, epochs: int, seed: int) -> None:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise SystemExit("Install TensorFlow first: pip install tensorflow") from error
    if not TRACE_PATH.exists():
        raise FileNotFoundError(f"Expected {TRACE_PATH}")

    tf.keras.utils.set_random_seed(seed)
    sequences, labels = reservoir_sample(per_class, seed)
    labels_array = np.array(labels)
    vocabulary_size = max(max(sequence) for sequence in sequences) + 1
    x_train_all, x_test, y_train_all, y_test = train_test_split(
        sequences, labels_array, test_size=0.20, random_state=seed, stratify=labels_array
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_all, y_train_all, test_size=0.25, random_state=seed, stratify=y_train_all
    )
    pad = tf.keras.utils.pad_sequences
    x_train = pad(x_train, maxlen=max_length, padding="post", truncating="post")
    x_validation = pad(x_validation, maxlen=max_length, padding="post", truncating="post")
    x_test = pad(x_test, maxlen=max_length, padding="post", truncating="post")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_length,)),
        tf.keras.layers.Embedding(vocabulary_size, 32, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.2)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile("adam", "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
    model.fit(
        x_train, y_train, validation_data=(x_validation, y_validation), epochs=epochs,
        batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=3, restore_best_weights=True)],
        verbose=2,
    )
    validation_scores = model.predict(x_validation, verbose=0).ravel()
    thresholds = np.linspace(0.05, 0.95, 91)
    threshold = max(thresholds, key=lambda value: f1_score(y_validation, validation_scores >= value))
    test_scores = model.predict(x_test, verbose=0).ravel()
    predictions = (test_scores >= threshold).astype(int)
    print(f"\nLSTM test report (threshold={threshold:.2f}):\n")
    print(classification_report(y_test, predictions, target_names=["Normal", "Anomaly"], digits=3))
    print(f"Anomaly F1 Score: {f1_score(y_test, predictions):.3f}")
    print(f"PR-AUC Score: {average_precision_score(y_test, test_scores):.3f}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACT_DIR / "hdfs_lstm.keras")
    (ARTIFACT_DIR / "hdfs_lstm_metadata.json").write_text(json.dumps({
        "dataset": "LogHub HDFS_v1 Event_traces.csv", "per_class": per_class,
        "max_length": max_length, "vocabulary_size": vocabulary_size,
        "selected_validation_threshold": float(threshold), "model": "Embedding + Bidirectional LSTM",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    main(args.per_class, args.max_length, args.epochs, args.seed)
