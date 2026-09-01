"""Extract the stance-detection subset from the UniAffect unified archive.

Usage:
    python scripts/prepare_data.py                 # demo subset (fast)
    python scripts/prepare_data.py --full          # entire stance subset
    python scripts/prepare_data.py --archive <path>

The script writes a self-contained dataset under data/stance/ so the rest of
the pipeline can run with paths relative to the project root.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = PROJECT_ROOT.parent / "UniAffect_unified_dataset.tar.gz"
JSONL_MEMBER = "data/unified_v2/by_tag/stance.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare the stance-detection dataset.")
    p.add_argument("--archive", default=str(DEFAULT_ARCHIVE), help="Path to the .tar.gz archive.")
    p.add_argument("--out", default=str(PROJECT_ROOT / "data" / "stance"), help="Output directory.")
    p.add_argument("--full", action="store_true", help="Extract the whole stance subset.")
    p.add_argument("--per-dataset-split", type=int, default=6,
                   help="Max samples per (dataset, split) when not --full.")
    p.add_argument("--image-source", default=None,
                   help="Optional directory of already-extracted images (copies instead of extracting).")
    return p.parse_args()


def load_records(tf: tarfile.TarFile) -> list:
    member = tf.getmember(JSONL_MEMBER)
    fh = tf.extractfile(member)
    if fh is None:
        raise RuntimeError(f"Could not open {JSONL_MEMBER} in the archive.")
    records = [json.loads(line) for line in fh.read().decode("utf-8").splitlines() if line.strip()]
    return records


def select(records: list, full: bool, per_ds: int) -> list:
    if full:
        return records
    # Keep the demo small but balanced across datasets, splits, and labels.
    from collections import defaultdict

    buckets = defaultdict(list)
    for rec in records:
        buckets[(rec["source_dataset"], rec["split"])].append(rec)

    final = []
    for recs in buckets.values():
        by_label = defaultdict(list)
        for rec in recs:
            by_label[rec["label"]].append(rec)
        labels = list(by_label.keys())
        order = []
        for i in range(max(len(v) for v in by_label.values())):
            for lab in labels:
                if i < len(by_label[lab]):
                    order.append(by_label[lab][i])
        final.extend(order[:per_ds])
    return final


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    archive = Path(args.archive)
    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as tf:
        records = load_records(tf)
        selected = select(records, args.full, args.per_dataset_split)
        needed_images = sorted({r["image_path"] for r in selected})

        # Write annotations with a repo-relative image path.
        ann_path = out_dir / "annotations.jsonl"
        with open(ann_path, "w", encoding="utf-8") as ann:
            for rec in selected:
                meta = dict(rec.get("meta", {}))
                meta.setdefault("stance_target", meta.get("stance_target", ""))
                new_rec = {
                    "sample_id": rec.get("sample_id"),
                    "text": rec.get("text"),
                    "image_path": f"images/{Path(rec.get('image_path', '')).name}",
                    "label": rec.get("label") or rec.get("gold_label"),
                    "gold_label": rec.get("gold_label"),
                    "candidate_labels": rec.get("candidate_labels", []),
                    "source_dataset": rec.get("source_dataset"),
                    "split": rec.get("split"),
                    "meta": meta,
                }
                ann.write(json.dumps(new_rec, ensure_ascii=False) + "\n")

        written = 0
        if args.image_source:
            src_dir = Path(args.image_source)
            for image_path in needed_images:
                src = src_dir / Path(image_path).name
                if src.exists():
                    (out_dir / "images" / src.name).write_bytes(src.read_bytes())
                    written += 1
        else:
            name_to_member = {Path(m.name).name: m for m in tf.getmembers() if m.isfile()}
            for image_path in needed_images:
                member = name_to_member.get(Path(image_path).name)
                if member is None:
                    continue
                blob = tf.extractfile(member)
                if blob is None:
                    continue
                (out_dir / "images" / Path(image_path).name).write_bytes(blob.read())
                written += 1

    logging.info("Wrote %d samples to %s", len(selected), ann_path)
    logging.info("Extracted %d images to %s/images", written, out_dir)


if __name__ == "__main__":
    sys.exit(main())
