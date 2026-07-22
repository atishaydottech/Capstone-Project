# Student GrantMatch

Agentic RAG system that tells students which scholarships and grants they actually qualify for, citing the exact rule behind each answer.

DEMO: https://www.loom.com/share/52453dab43bd488ca2402a83a92e0150

**Group:** G4, GardenState.AI
**Team:** Atishay Jain, Dinesh Chandramohan

## Problem

Students often don't know which scholarships or grants they actually qualify for. Rules are scattered across long documents, and keyword search misses matches or gives wrong ones. This tool takes a student's basic info and tells them which programs they qualify for, citing the exact rule that says so.

## Goals

- Check a student's info (income, state, GPA, major, etc.) against real program rules.
- Show the exact rule behind each yes or no, not just an answer.
- Flag partial matches instead of forcing a plain yes or no.
- Search documents only when needed, not on every query.
- Test against labeled profiles and measure real accuracy.
- Report eligibility only, not odds of winning.

## Architecture

**Offline ingestion:** program rule documents are chunked, embedded (OpenAI), and stored in ChromaDB.

**Runtime (per query):** a ReAct agent (Pydantic AI + OpenAI) takes a student query, decides whether to call the retriever tool or the web search tool, observes the result, loops if needed, and returns a final answer with a cited clause.

```
Docs -> Chunk + Embed (OpenAI) -> ChromaDB
                                     |
User -> Frontend -> ReAct Agent (Pydantic AI + OpenAI)
                       |     \
                 Retriever   Web Search
                    Tool        Tool
                       \        /
                     Final Answer -> Frontend
```

## Tool Stack

- **Pydantic AI** - agent loop and tool-calling (Thought / Action / Observation), plus structured, validated outputs.
- **OpenAI API** - LLM for reasoning + embeddings (text-embedding-3-small).
- **ChromaDB** - local vector store for retrieved chunks.
- **Retriever tool** - wraps the vector store as a callable agent tool.
- **Web search tool** (Tavily / SerpAPI) - second tool, gives the agent a real choice.

Pydantic AI was chosen over LangGraph because this project's loop (retrieve, maybe search, answer) doesn't need complex branching, and Pydantic AI's structured output format directly supports the eligibility + citation format this project needs, without a second framework on top.

## Evaluation Criteria

**Quantitative**
- Eligibility accuracy vs. 25-30 hand-labeled student profiles.
- Citation accuracy: correct rule quoted, not just a correct answer.
- Hallucination rate: cited details not actually in the source.
- Tool-call count and response latency per query.
- Consistency across repeated identical queries.
- Targets: >80% eligibility accuracy, >70% citation accuracy, <10% hallucination rate.

**Qualitative**
- Rate cited rules as correct, close, or unrelated on a sample.
- Review worst-performing cases and tag the failure cause.
- Check answers are understandable without reading the source document.
- Check tone is neutral and appropriately cautious on uncertain cases.
- Manually review borderline/edge-case profiles for nuance.

## Failure Analysis

Real pivots hit during the build, pulled from the commit history:

- **ChromaStore was wiping its own data on every reopen.** The original store deleted and rebuilt the collection every time it was instantiated, so any query-time use of it (the agent, the app) erased what `ingest.py` had just built. Fixed by adding a reset flag: the default is now to reopen an existing collection, and only `python ingest.py` with `reset=True` rebuilds it on purpose.
- **We assumed the wrong model family.** The design called for `gpt-4o` and `gpt-4o-mini`, but this project's OpenAI key only has access to the `gpt-5.4` family. That showed up as a 403 `model_not_found` error in practice, not something caught at design time. `shared/llm.py` and `agent/agent.py` were switched to `gpt-5.4-mini`, `gpt-5.4-nano`, and `text-embedding-3-small`.
- **Streamlit Cloud deployment added complexity without paying off.** Getting Streamlit's secrets to reach the environment before `agent.py` builds its agent at import time needed a bridge in `app.py`, since `.env` is gitignored and never reaches Cloud. Once we decided the demo would run locally, we removed that bridge and the dev container files. Confirmed the app still starts clean and serves fine without them.
- **A Streamlit session-state bug.** Setting `st.session_state.query_input` after the bound text box was already created raised an exception every time someone clicked an example question, and separately caused typed input to silently snap back to the last example on rerun. Fixed by moving the button handling above the text box, so state is set before the widget exists, and binding the box directly to session state instead of reading it with a fallback default.
- **Hybrid retrieval was built but not adopted into the agent's retriever tool.** `retrieval/hybrid.py` and `bm25_search.py` (dense search plus BM25 via Reciprocal Rank Fusion) are wired into the standalone `rag/pipeline.py` baseline, but at this corpus size (34 chunks from 10 programs) plain vector search isn't a bottleneck, so the MVP scope deliberately left hybrid search, reranking, and query rewriting for later.
- **The agent loop was first tested against a fake store.** It was built and schema-validated before `pydantic-ai` was even installed in the build environment, so the full loop was unverified end to end for a while. That got closed out once the real dependency was installed and a live run returned a correctly grounded answer citing the actual Cal Grant A clause.

## Setup

```bash
git clone https://github.com/atishaydottech/Capstone-Project
cd Capstone-Project
pip install -r requirements.txt
```

Add your OpenAI API key (and web search API key) to a `.env` file:

```
OPENAI_API_KEY=your_key_here
SEARCH_API_KEY=your_key_here
```

Run ingestion, then start the app:

```bash
python ingest.py
streamlit run app.py
```

## Status

Code complete: ingestion, the agent (both tools, typed output, guardrails, tracing), the vanilla RAG baseline, the eval harness, and the UI are all built and in this repo.

`eval/golden_profiles.json` has 30 hand-labeled profiles across all 10 programs. What's still open: running `eval/run_eval.py` and the baseline against that set with real API access, filling in the eligibility/citation/hallucination numbers, and writing a short conclusion on which approach wins and why.
