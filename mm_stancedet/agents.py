"""Specialised agents that make up the multimodal analysis and debate stages."""

from __future__ import annotations

from typing import Optional

from . import prompts
from .llm_client import LLMClient
from .schema import StanceSample, parse_adjudicator_output


class BaseAgent:
    def __init__(self, client: LLMClient):
        self.client = client


class TextAnalysisAgent(BaseAgent):
    """Extract linguistic features relevant to the target."""

    def run(self, sample: StanceSample) -> str:
        system, user = prompts.text_analysis_prompt(sample.text, sample.target)
        return self.client.complete(system, user)


class ImageAnalysisAgent(BaseAgent):
    """Interpret visual cues relevant to the target."""

    def run(self, sample: StanceSample, image_path: str) -> str:
        system, user = prompts.image_analysis_prompt(sample.target)
        return self.client.complete(system, user, image_path=image_path)


class ModalityConflictAgent(BaseAgent):
    """Assess alignment / divergence between text and image."""

    def run(self, sample: StanceSample, image_path: str, exemplar_info: str) -> str:
        system, user = prompts.modality_conflict_prompt(
            sample.text, sample.target, exemplar_info
        )
        return self.client.complete(system, user, image_path=image_path)


class DebaterAgent(BaseAgent):
    """Argue in favour of a specific stance label."""

    def __init__(self, client: LLMClient, stance: str):
        super().__init__(client)
        self.stance = stance

    def run(
        self,
        sample: StanceSample,
        text_analysis: str,
        image_analysis: str,
        conflict_analysis: str,
        debate_context: Optional[str],
    ) -> str:
        system, user = prompts.debater_prompt(
            self.stance,
            sample.text,
            sample.target,
            text_analysis,
            image_analysis,
            conflict_analysis,
            debate_context,
        )
        return self.client.complete(system, user)


class AdjudicatorAgent(BaseAgent):
    """Meta-reasoner that reflects on the debate and issues the final stance."""

    def run(
        self,
        sample: StanceSample,
        text_analysis: str,
        image_analysis: str,
        conflict_analysis: str,
        arguments: dict,
    ) -> tuple[str, str]:
        system, user = prompts.adjudicator_prompt(
            sample.text,
            sample.target,
            text_analysis,
            image_analysis,
            conflict_analysis,
            arguments,
        )
        output = self.client.complete(system, user)
        return parse_adjudicator_output(output)

