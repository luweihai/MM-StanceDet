"""MM-StanceDet orchestration: the four-stage agentic pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .agents import (
    AdjudicatorAgent,
    DebaterAgent,
    ImageAnalysisAgent,
    ModalityConflictAgent,
    TextAnalysisAgent,
)
from .config import Config
from .llm_client import LLMClient
from .retrieval import Retriever
from .schema import STANCE_LABELS, StanceSample


logger = logging.getLogger(__name__)


class MMStanceDet:
    """Four-stage framework for multimodal stance detection."""

    def __init__(
        self,
        cfg: Config,
        client: Optional[LLMClient] = None,
        retriever: Optional[Retriever] = None,
    ):
        self.cfg = cfg
        self._image_resolver: Callable[[StanceSample], str] = self._make_image_resolver()
        self.client = client or LLMClient(cfg.llm, cfg.api_key())
        self.retriever = retriever or Retriever(cfg, image_resolver=self._image_resolver)

        # Stage-2 specialised agents.
        self.text_agent = TextAnalysisAgent(self.client)
        self.image_agent = ImageAnalysisAgent(self.client)
        self.conflict_agent = ModalityConflictAgent(self.client)

        # Stage-3 debater agents, one per possible stance.
        self.debaters: Dict[str, DebaterAgent] = {
            label: DebaterAgent(self.client, label) for label in STANCE_LABELS
        }

        # Stage-4 adjudicator.
        self.adjudicator = AdjudicatorAgent(self.client)
        self._pool: List[StanceSample] = []

    def _make_image_resolver(self) -> Callable[[StanceSample], str]:
        def resolve(sample: StanceSample) -> str:
            rel = Path(self.cfg.data.root) / sample.image_path
            return str(self.cfg.resolve(rel))

        return resolve

    def build_retrieval_index(self, pool: List[StanceSample]) -> None:
        self._pool = pool
        self.retriever.build(pool)

    def _format_exemplars(self, exemplars: List[StanceSample]) -> str:
        if not exemplars:
            return "No contextual examples retrieved."
        parts = []
        for i, ex in enumerate(exemplars, 1):
            cot = ex.meta.get("cot", "") if isinstance(ex.meta, dict) else ""
            reason = cot or (f"The stance towards '{ex.target}' is {ex.label}.")
            parts.append(
                f"Example {i} (Text: \"{ex.text[:160]}\"; Target: \"{ex.target}\"; "
                f"Stance: \"{ex.label}\"; Reasoning: {reason})"
            )
        return "\n".join(parts)

    @staticmethod
    def _format_debate_context(arguments: Dict[str, str]) -> str:
        lines = []
        for stance, arg in arguments.items():
            if arg:
                lines.append(f"[{stance} argument]\n{arg}")
        return "\n\n".join(lines) if lines else "No previous arguments yet."

    def predict(self, sample: StanceSample) -> tuple[str, str, dict]:
        """Run the full pipeline and return (label, justification, trace)."""
        if not self._pool:
            raise RuntimeError("Retrieval index is empty. Call build_retrieval_index first.")

        image_abs = self._image_resolver(sample)

        # Stage 1: retrieval augmentation.
        hits = self.retriever.search(sample, self.cfg.retrieval.top_k)
        exemplars = [self._pool[idx] for _, idx in hits]
        exemplar_info = self._format_exemplars(exemplars)

        # Stage 2: multimodal analysis.
        text_analysis = self.text_agent.run(sample)
        image_analysis = self.image_agent.run(sample, image_abs)
        conflict_analysis = self.conflict_agent.run(sample, image_abs, exemplar_info)

        # Stage 3: reasoning-enhanced debate over 'rounds' iterations.
        arguments: Dict[str, str] = {label: "" for label in STANCE_LABELS}
        for _ in range(self.cfg.debate.rounds):
            context = self._format_debate_context(arguments)
            for label in STANCE_LABELS:
                arguments[label] = self.debaters[label].run(
                    sample, text_analysis, image_analysis, conflict_analysis, context
                )

        # Stage 4: self-reflection and adjudication.
        label, justification = self.adjudicator.run(
            sample, text_analysis, image_analysis, conflict_analysis, arguments
        )

        trace = {
            "exemplars": [[e.sample_id, e.source_dataset, e.label] for e in exemplars],
            "text_analysis": text_analysis,
            "image_analysis": image_analysis,
            "conflict_analysis": conflict_analysis,
            "arguments": arguments,
            "justification": justification,
        }
        return label, justification, trace

