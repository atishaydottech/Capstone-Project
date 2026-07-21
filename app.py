"""Streamlit UI — reference-only in spirit from single-agent-lab/app.py (which
is Gradio); this is a fresh Streamlit build per the design doc, not a port.

Left: the student's question. Right: the validated typed answer. Below: the
ReAct trace (Thought / Action / Observation), so the retriever-vs-web-search
choice and the citation are visible, not a black box.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st

# Streamlit Community Cloud doesn't read .env (gitignored, not deployed) --
# it exposes secrets via st.secrets instead. Bridge them into the environment
# BEFORE importing agent.trace, since agent/agent.py builds the Agent (and
# reads OPENAI_API_KEY) at import time. Locally, .env via load_dotenv already
# populated these, and st.secrets is just empty, so this is a no-op there.
try:
    _secrets = dict(st.secrets)
except Exception:
    _secrets = {}  # no secrets.toml at all (e.g. local dev without Streamlit secrets) -- fine
for _key in ("OPENAI_API_KEY", "SEARCH_API_KEY", "AGENT_MODEL"):
    if _key not in os.environ and _key in _secrets:
        os.environ[_key] = _secrets[_key]

from agent.schemas import EligibilityAnswer, NeedMoreInfo
from agent.trace import run

ICON = {"thought": "💭", "action": "🔧", "observation": "📡", "retry": "↻", "final": "✅"}

EXAMPLES = [
    "I have a 3.4 GPA, live in California, and this is my first undergraduate degree. Am I eligible for Cal Grant A?",
    "I've lived in New Jersey for 3 years, enrolled full-time, household income $50,000. Am I eligible for NJ TAG?",
    "I graduated from a Georgia high school with a 3.2 GPA. Am I eligible for the HOPE Scholarship?",
    "Am I eligible for a scholarship?",
]


def render_answer(output):
    if isinstance(output, EligibilityAnswer):
        badge = {"yes": "🟢 Eligible", "no": "🔴 Not eligible", "partial": "🟡 Partially eligible"}[output.eligible]
        st.markdown(f"### {badge} — {output.program}")
        st.markdown(f"**Cited clause** · {output.cited_clause}")
        st.caption(f"Source: {output.cited_source}")
        st.markdown(f"**Reasoning** · {output.reasoning}")
        if output.caveats:
            st.markdown("**Caveats**")
            for c in output.caveats:
                st.markdown(f"- {c}")
    elif isinstance(output, NeedMoreInfo):
        st.markdown("### 🤔 Need more info")
        st.markdown(f"**Question** · {output.question}")
        st.caption(f"Why: {output.reason}")
    else:
        st.code(repr(output))


def render_trace(steps, tool_sequence):
    for s in steps:
        icon = ICON.get(s.kind, "•")
        with st.container(border=True):
            st.markdown(f"**{icon} {s.title}**")
            st.text(s.body)
    st.caption("Tool sequence: " + (" → ".join(tool_sequence) or "(no tools called)"))


def main():
    st.set_page_config(page_title="GrantMatch", page_icon="🎓", layout="wide")
    st.title("🎓 GrantMatch")
    st.caption("Tells students which scholarships and grants they actually qualify for, citing the exact rule behind each answer.")

    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    # Buttons must be handled BEFORE the bound text_area is instantiated below --
    # Streamlit forbids setting session_state[key] after that key's widget already
    # exists in the current run.
    st.write("Try one:")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example[:40] + "...", use_container_width=True):
            st.session_state.query_input = example

    st.text_area("Describe your situation and ask about a program", height=100, key="query_input")

    query = st.session_state.query_input
    go = st.button("Ask", type="primary")

    if go and query.strip():
        with st.spinner("Checking eligibility..."):
            try:
                steps, output, tool_sequence, _ = run(query)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                return

        answer_col, trace_col = st.columns([2, 3])
        with answer_col:
            st.subheader("Answer")
            render_answer(output)
        with trace_col:
            st.subheader("Agent trace")
            render_trace(steps, tool_sequence)
    elif go:
        st.warning("Type a question first.")


if __name__ == "__main__":
    main()
