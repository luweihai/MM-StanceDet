"""Prompt templates for every agent in MM-StanceDet (see paper Appendix A.4)."""

from __future__ import annotations

from typing import Optional


def text_analysis_prompt(text: str, target: str) -> tuple[str, str]:
    system = (
        "You are a Text Analysis Agent. Your task is to analyze the given text to "
        "identify linguistic features relevant to determining the author's stance "
        "towards a specific target."
    )
    user = (
        "Input:\n"
        "  - Text: \"{text}\"\n"
        "  - Target: \"{target}\"\n\n"
        "Your analysis should include:\n"
        "  1. Keywords and salient phrases/sentences related to the target.\n"
        "  2. Explicit or implicit sentiment polarity towards the target.\n"
        "  3. Detection of potential sarcasm, irony, or subtle nuances.\n"
        "  4. Overall topic relevance concerning the target.\n\n"
        "Provide a structured analysis."
    ).format(text=text, target=target)
    return system, user


def image_analysis_prompt(target: str) -> tuple[str, str]:
    system = (
        "You are an Image Analysis Agent. Your task is to interpret the visual "
        "content of an image to find cues relevant to determining the author's "
        "stance towards a specific target."
    )
    user = (
        "Input:\n"
        "  - Image: (provided as input, analyze it)\n"
        "  - Target: \"{target}\"\n\n"
        "Your analysis should include:\n"
        "  1. Descriptions of relevant visual objects and their context.\n"
        "  2. Overall scene context and setting.\n"
        "  3. Inferred emotions from depicted individuals (if any).\n"
        "  4. Connotations suggested by color palettes, composition, or symbolism "
        "related to the target.\n\n"
        "Provide a structured visual analysis."
    ).format(target=target)
    return system, user


def modality_conflict_prompt(text: str, target: str, exemplar_info: str) -> tuple[str, str]:
    system = (
        "You are a Modality Conflict Agent. Your primary function is to assess the "
        "interplay between the provided image and text concerning the target. Detect "
        "potential inconsistencies, contradictions, or synergistic reinforcements "
        "between the modalities."
    )
    user = (
        "Input:\n"
        "  - Image: (provided as input, analyze it)\n"
        "  - Text: \"{text}\"\n"
        "  - Target: \"{target}\"\n"
        "  - exemplar_info: {exemplar_info}\n\n"
        "Your assessment should:\n"
        "  1. Highlight specific conflicting signals (e.g., text favors but image againsts).\n"
        "  2. Highlight specific reinforcing cues (e.g., both text and image strongly favor).\n"
        "  3. Explain how the modalities align or diverge in expressing a stance towards "
        "the target \"{target}\".\n"
        "  4. Reference patterns or reasoning observed in the provided contextual examples "
        "if they are relevant.\n\n"
        "Provide a detailed assessment of inter-modal alignment or divergence."
    ).format(text=text, target=target, exemplar_info=exemplar_info)
    return system, user


def debater_prompt(
    stance: str,
    text: str,
    target: str,
    text_analysis: str,
    image_analysis: str,
    conflict_analysis: str,
    debate_context: Optional[str],
) -> tuple[str, str]:
    system = (
        "You are a Debater Agent arguing for the '{stance}' stance. Your goal is to "
        "construct a coherent argument, synthesizing all provided information, to explain "
        "why the given multimodal instance expresses a '{stance}' stance towards the target."
    ).format(stance=stance)
    ctx = debate_context if debate_context else "No previous arguments yet."
    user = (
        "Input Instance:\n"
        "  - Text: \"{text}\"\n"
        "  - Target: \"{target}\"\n\n"
        "Analysis Results:\n"
        "  - Text Analysis: {text_analysis}\n"
        "  - Image Analysis: {image_analysis}\n"
        "  - Modality Conflict Analysis: {conflict_analysis}\n"
        "  - debate_context: {ctx}\n\n"
        "Construct your argument. Clearly reference details from the text, image "
        "analysis, and modality conflict analysis to favor your position. If previous "
        "arguments from other debaters are provided, aim to strengthen your argument in "
        "light of their points, but focus on building your case. Do not explicitly state "
        "\"I am arguing for...\". Just present the argument."
    ).format(
        text=text,
        target=target,
        text_analysis=text_analysis,
        image_analysis=image_analysis,
        conflict_analysis=conflict_analysis,
        ctx=ctx,
    )
    return system, user


def adjudicator_prompt(
    text: str,
    target: str,
    text_analysis: str,
    image_analysis: str,
    conflict_analysis: str,
    args: dict,
) -> tuple[str, str]:
    system = (
        "You are an Adjudicator Agent. Your task is to critically evaluate competing "
        "arguments and comprehensive analyses to determine the definitive stance "
        "(Favor, Neutral, or Against) expressed in a multimodal instance towards a "
        "specific target."
    )
    user = (
        "Input Instance:\n"
        "  - Text: \"{text}\"\n"
        "  - Target: \"{target}\"\n\n"
        "Analysis Results:\n"
        "  - Text Analysis: {text_analysis}\n"
        "  - Image Analysis: {image_analysis}\n"
        "  - Modality Conflict Analysis: {conflict_analysis}\n\n"
        "Arguments from Debater Agents:\n"
        "  - Favor Argument: {favor_arg}\n"
        "  - Against Argument: {against_arg}\n"
        "  - Neutral Argument: {neutral_arg}\n\n"
        "Perform the following steps:\n"
        "  1. Initial Assessment: Briefly summarize the strengths and weaknesses of "
        "each argument based on the provided analyses.\n"
        "  2. Critical Self-Reflection: Actively look for inconsistencies, overlooked "
        "modality conflicts (referencing Modality Conflict Analysis), or weak reasoning "
        "points.\n"
        "  3. Final Decision: Based on your comprehensive evaluation and critical "
        "self-reflection, determine the most justified stance.\n"
        "  4. Justification: Provide a clear, concise justification for your final "
        "decision, incorporating insights from your self-reflection.\n\n"
        "Your output format should be:\n"
        "Stance: [Favor|Neutral|Against]\n"
        "Justification: [Your detailed reasoning]"
    ).format(
        text=text,
        target=target,
        text_analysis=text_analysis,
        image_analysis=image_analysis,
        conflict_analysis=conflict_analysis,
        favor_arg=args.get("Support", ""),
        against_arg=args.get("Oppose", ""),
        neutral_arg=args.get("Neutral", ""),
    )
    return system, user

