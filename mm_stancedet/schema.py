"""Data schema and label mapping shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Canonical label set used throughout the framework (matches the dataset).
STANCE_LABELS: List[str] = ["Support", "Oppose", "Neutral"]

# Numeric mapping used for evaluation: Support=1, Oppose=-1, Neutral=0.
_LABEL_TO_IDX: Dict[str, int] = {"Support": 1, "Oppose": -1, "Neutral": 0}
_IDX_TO_LABEL: Dict[int, str] = {v: k for k, v in _LABEL_TO_IDX.items()}


def label_to_index(label: str) -> int:
    return _LABEL_TO_IDX[label]


def index_to_label(idx: int) -> str:
    return _IDX_TO_LABEL[idx]


def normalize_label(label: str) -> str:
    """Map the many spellings used in logs/LLM output to the canonical set."""
    token = label.strip().lower()
    if token in {"support", "favor", "favour", "in favor", "supporting"}:
        return "Support"
    if token in {"oppose", "against", "against/oppose", "opposition"}:
        return "Oppose"
    if token in {"neutral", "none", "unrelated", "comment", "impartial"}:
        return "Neutral"
    raise ValueError(f"Unknown stance label: {label!r}")


@dataclass
class StanceSample:
    """A single multimodal stance instance."""

    sample_id: str
    text: str
    image_path: str
    target: str
    label: Optional[str] = None
    candidate_labels: List[str] = field(default_factory=lambda: list(STANCE_LABELS))
    source_dataset: str = ""
    split: str = ""
    meta: Dict = field(default_factory=dict)

    @property
    def label_index(self) -> Optional[int]:
        return label_to_index(self.label) if self.label else None

    def key(self) -> str:
        """Hashable identity for retrieval bookkeeping."""
        return self.sample_id


def parse_adjudicator_output(output: str) -> tuple[str, str]:
    """Parse the Adjudicator's fixed-format response into (label, justification)."""
    stance = "Neutral"
    justification = ""
    for line in output.splitlines():
        line = line.strip()
        if line.lower().startswith("stance:"):
            stance = normalize_label(line.split(":", 1)[1])
        elif line.lower().startswith("justification:"):
            justification = line.split(":", 1)[1].strip()
    if not justification:
        justification = output.strip()
    return stance, justification

