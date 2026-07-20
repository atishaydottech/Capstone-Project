"""Typed contracts for the GrantMatch agent.

Mirrors single-agent-lab's TravelBriefing / NeedMoreInfo union: the agent's
final answer is either a grounded EligibilityAnswer or an admission that it
needs more information. Schema validation (Pydantic) rejects a malformed
answer before it ever reaches the student; business-logic checks (e.g. a
citation that's actually empty) are caught separately in agent.py's
@agent.output_validator.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EligibilityAnswer(BaseModel):
    """A grounded eligibility answer. cited_clause/cited_source must trace
    back to a real retriever_tool or web_search result — never invented."""

    program: str = Field(description="The scholarship or grant program this answer is about")
    eligible: Literal["yes", "no", "partial"] = Field(
        description="'partial' when the student meets some but not all requirements"
    )
    cited_clause: str = Field(description="The exact rule text the answer relies on")
    cited_source: str = Field(description="Program name / document the clause came from")
    reasoning: str = Field(description="How the student's info was checked against the cited clause")
    caveats: list[str] = Field(default_factory=list, description="Anything the rules didn't fully cover")


class NeedMoreInfo(BaseModel):
    """Return this INSTEAD of an EligibilityAnswer when eligibility can't be
    determined — missing student info, or no matching program found by
    either tool. Say so and ask, rather than guessing."""

    question: str = Field(description="The one thing needed from the student to proceed")
    reason: str = Field(description="Why eligibility could not be determined from tool results alone")
