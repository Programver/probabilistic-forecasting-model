"""
Insight orchestration: forecast -> grounded drivers -> narrative (+ optional LLM).

One entry point (:func:`generate_briefing`) that the app and any CLI use. It always
returns a complete briefing; the LLM only ever *upgrades* the offline draft.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from drivers import DriverReport, build_driver_report
from llm import generate as llm_generate
from logging_utils import get_logger
from narrative import build_sections, to_markdown

log = get_logger("insights")


@dataclass
class Briefing:
    report: DriverReport
    sections: Dict[str, str]
    offline_markdown: str
    markdown: str
    provider: str  # "offline" | "gemini" | "openai" | "anthropic"
    model: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider, "model": self.model,
            "markdown": self.markdown, "sections": self.sections,
            "report": self.report.to_dict(),
        }


def generate_briefing(result, prepared, model, use_llm: bool = True) -> Briefing:
    report = build_driver_report(result, prepared, model)
    sections = build_sections(report)
    draft = to_markdown(report)

    provider, model_name, markdown = "offline", None, draft
    if use_llm:
        res = llm_generate(report, draft)
        if res is not None and res.text.strip():
            provider, model_name, markdown = res.provider, res.model, res.text
            log.info("insight briefing polished by %s (%s)", provider, model_name)

    return Briefing(report=report, sections=sections, offline_markdown=draft,
                    markdown=markdown, provider=provider, model=model_name)
