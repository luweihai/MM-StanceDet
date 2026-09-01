"""Compute Macro-F1 over the prediction file produced by run.py."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mm_stancedet.schema import STANCE_LABELS, label_to_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/predictions.jsonl")
    parser.add_argument("--gold-field", default="gold_label")
    parser.add_argument("--pred-field", default="pred_label")
    args = parser.parse_args()

    y_true, y_pred = [], []
    with open(args.predictions, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            y_true.append(label_to_index(rec[args.gold_field]))
            y_pred.append(label_to_index(rec[args.pred_field]))

    labels = [label_to_index(lbl) for lbl in STANCE_LABELS]
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"Samples: {len(y_true)}  Accuracy: {acc:.4f}  Macro-F1: {macro:.4f}")


if __name__ == "__main__":
    sys.exit(main())

