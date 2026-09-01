"""End-to-end inference entry point for MM-StanceDet.

Examples:
    python scripts/run.py                      # offline smoke test on a small subset
    python scripts/run.py --offline --limit 8
    python scripts/run.py --limit 20 --from-api
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mm_stancedet import MMStanceDet, load_config, load_samples  # noqa: E402
from mm_stancedet.llm_client import LLMClient  # noqa: E402
from mm_stancedet.retrieval import Retriever, SummaryEncoder  # noqa: E402


class OfflineResponder(LLMClient):
    """Deterministic responder used for network-free smoke tests of the pipeline."""

    def complete(self, system, user_text, image_path=None, temperature=None):
        if system.startswith("You are an Adjudicator"):
            guess = self._guess(user_text)
            return f"Stance: {guess}\nJustification: Offline deterministic check."
        if system.startswith("You are a Debater"):
            return "The textual and visual evidence point consistently to this stance."
        if system.startswith("You are a Modality Conflict"):
            return "The image and text align with no strong conflicting signal."
        if system.startswith("You are an Image Analysis"):
            return "Visual content is neutral with no explicit stance cue."
        return "Concise linguistic analysis of the target."

    @staticmethod
    def _guess(text: str) -> str:
        low = text.lower()
        if any(w in low for w in ["support", "great", "love", "thank", "proud", "yes"]):
            return "Support"
        if any(w in low for w in ["against", "hate", "oppose", "never", "no", "fail", "wrong"]):
            return "Oppose"
        return "Neutral"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MM-StanceDet inference.")
    parser.add_argument("--offline", action="store_true", help="Use deterministic responders (no API).")
    parser.add_argument("--limit", type=int, default=8, help="Number of eval samples to process.")
    parser.add_argument("--save-trace", action="store_true", help="Dump the full intermediate trace.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config()
    groups = load_samples(cfg)

    pool = groups.get(cfg.data.retrieval_split, [])
    eval_samples = groups.get(cfg.data.eval_split, [])[: args.limit]
    logging.info("Reference pool: %d samples; evaluation set: %d samples.", len(pool), len(eval_samples))

    if args.offline:
        client = OfflineResponder(cfg.llm, api_key="offline")
        retriever = Retriever(cfg, encoder=SummaryEncoder(cfg.retrieval.embedding_dim),
                              image_resolver=lambda s: cfg.resolve(Path(cfg.data.root) / s.image_path))
        model = MMStanceDet(cfg, client=client, retriever=retriever)
    else:
        model = MMStanceDet(cfg)

    model.build_retrieval_index(pool)

    out_dir = cfg.resolve(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions.jsonl"
    with open(out_file, "w", encoding="utf-8") as out:
        for sample in eval_samples:
            pred_label, justification, trace = model.predict(sample)
            record = {
                "sample_id": sample.sample_id,
                "source_dataset": sample.source_dataset,
                "split": sample.split,
                "target": sample.target,
                "gold_label": sample.label,
                "pred_label": pred_label,
                "justification": justification,
            }
            if args.save_trace:
                record["trace"] = trace
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            logging.info("[%s] gold=%s pred=%s", sample.sample_id, sample.label, pred_label)

    logging.info("Predictions written to %s", out_file)


if __name__ == "__main__":
    sys.exit(main())

