"""Run the agent and turn its message history into a ReAct trace: Thought,
Action, Observation, Final. Pydantic AI hands back the whole history via
result.all_messages() -- this just walks it and labels the parts."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import LIMITS, agent
from agent.schemas import NeedMoreInfo

# Guardrail / loop-limit exceptions we degrade gracefully instead of crashing on.
try:
    from pydantic_ai import UsageLimitExceeded
except Exception:  # pragma: no cover - import path varies by version
    from pydantic_ai.usage import UsageLimitExceeded  # type: ignore
try:
    from pydantic_ai.exceptions import UnexpectedModelBehavior
except Exception:  # pragma: no cover
    UnexpectedModelBehavior = ()  # type: ignore

GUARDRAIL_ERRORS = tuple(e for e in (UsageLimitExceeded, UnexpectedModelBehavior) if isinstance(e, type))


@dataclass
class Step:
    kind: str  # thought | action | observation | retry | final | error
    title: str
    body: str


def _args_str(part) -> str:
    """ToolCallPart.args may be a dict or a JSON string across versions."""
    if hasattr(part, "args_as_json_str"):
        try:
            return part.args_as_json_str()
        except Exception:
            pass
    args = getattr(part, "args", None)
    if isinstance(args, (dict, list)):
        return json.dumps(args)
    return str(args)


def run(query: str) -> tuple[list[Step], object, list[str], list]:
    """Execute one agent run.

    Returns (trace steps, final output object, ordered list of tool names called,
    raw messages as JSON-able Python). The tool-name sequence is what the eval
    scores; the raw messages power the "Raw JSON" view.
    """
    try:
        result = agent.run_sync(query, usage_limits=LIMITS)
    except GUARDRAIL_ERRORS as e:
        # A guardrail (UsageLimits) or an exhausted retry budget fired. That is
        # the guardrail working — bound the blast radius. Degrade gracefully to
        # an abstention instead of crashing the app.
        out = NeedMoreInfo(
            question="Could you share your GPA, state, and income, and the program you're asking about?",
            reason=f"I couldn't determine eligibility and stopped to avoid looping ({type(e).__name__}).",
        )
        return [Step("retry", "Guardrail stopped the run", str(e)),
                Step("final", "Final answer", repr(out))], out, [], []

    steps: list[Step] = []
    tool_sequence: list[str] = []

    for message in result.all_messages():
        for part in getattr(message, "parts", []):
            name = type(part).__name__
            if name in ("TextPart", "ThinkingPart"):
                body = (getattr(part, "content", "") or "").strip()
                if body:
                    steps.append(Step("thought", "Thought", body))
            elif name == "ToolCallPart":
                # Pydantic AI emits the structured output as a synthetic
                # `final_result_<Type>` tool call — that's the answer, not a
                # real tool. Skip it here; the Final row below already shows it.
                if part.tool_name.startswith("final_result"):
                    continue
                steps.append(Step("action", "Action", f"{part.tool_name}({_args_str(part)})"))
                tool_sequence.append(part.tool_name)
            elif name == "ToolReturnPart":
                if part.tool_name.startswith("final_result"):
                    continue
                steps.append(Step("observation", "Observation", f"{part.tool_name} -> {part.content}"))
            elif name == "RetryPromptPart":
                body = getattr(part, "content", part)
                steps.append(Step("retry", "Retry (validation failed)", str(body)))

    output = getattr(result, "output", None)
    if output is None:
        output = getattr(result, "data", None)
    steps.append(Step("final", "Final answer", repr(output)))

    try:
        raw_messages = json.loads(result.all_messages_json())
    except Exception:
        raw_messages = []

    return steps, output, tool_sequence, raw_messages


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "I have a 3.2 GPA and live in California — do I qualify for Cal Grant A?"
    trace, output, seq, _ = run(q)
    for s in trace:
        print(f"\n[{s.title}]\n{s.body}")
    print("\nTOOL SEQUENCE:", " -> ".join(seq) or "(none)")
