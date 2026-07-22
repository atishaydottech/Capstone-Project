# The agent's final answer is either a grounded EligibilityAnswer or an
# admission that it needs more info -- see agent.py's output_type union.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EligibilityAnswer(BaseModel):
    # cited_clause/cited_source must trace back to a real tool result, not
    # something the model made up.
    program: str = Field(description="The scholarship or grant program this answer is about")
    eligible: Literal["yes", "no", "partial"] = Field(
        description="'partial' when the student meets some but not all requirements"
    )
    cited_clause: str = Field(description="The exact rule text the answer relies on")
    cited_source: str = Field(description="Program name / document the clause came from")
    reasoning: str = Field(description="How the student's info was checked against the cited clause")
    caveats: list[str] = Field(default_factory=list, description="Anything the rules didn't fully cover")


class NeedMoreInfo(BaseModel):
    # Returned instead of EligibilityAnswer when we genuinely can't tell --
    # missing info, or neither tool found the program.
    question: str = Field(description="The one thing needed from the student to proceed")
    reason: str = Field(description="Why eligibility could not be determined from tool results alone")
