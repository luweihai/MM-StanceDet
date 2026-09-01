"""Dataset loading for the stance-detection subset."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from .config import Config
from .schema import StanceSample


logger = logging.getLogger(__name__)


def _to_image_path(images_dir: Path, raw_path: str) -> str:
    """Normalise an image path to a path relative to the project data root."""
    name = Path(raw_path).name
    return f"{images_dir}/{name}"


def load_samples(cfg: Config) -> Dict[str, List[StanceSample]]:
    """Load all stance samples grouped by split (valid / test / train)."""
    annotations = cfg.resolve(Path(cfg.data.root) / cfg.data.annotations)
    images_dir = cfg.data.images_dir
    groups: Dict[str, List[StanceSample]] = {}
    with open(annotations, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("meta", {})
            sample = StanceSample(
                sample_id=rec.get("sample_id", ""),
                text=rec.get("text", ""),
                image_path=_to_image_path(images_dir, rec.get("image_path", "")),
                target=meta.get("stance_target", ""),
                label=rec.get("label") or rec.get("gold_label"),
                candidate_labels=rec.get("candidate_labels", []),
                source_dataset=rec.get("source_dataset", ""),
                split=rec.get("split", ""),
                meta=meta,
            )
            groups.setdefault(sample.split, []).append(sample)
    logger.info("Loaded %d stance samples across splits: %s", sum(len(v) for v in groups.values()),
                {k: len(v) for k, v in groups.items()})
    return groups


def build_reference_pool(cfg: Config, samples: Dict[str, List[StanceSample]],
                         split: str) -> List[StanceSample]:
    """Return the samples used to build the retrieval reference pool."""
    pool = samples.get(split, [])
    return pool

