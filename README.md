# Student GrantMatch

Agentic RAG system that tells students which scholarships and grants they actually qualify for, citing the exact rule behind each answer.

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
- **Streamlit** - simple front end for the demo.

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

In progress, 10-day capstone build.
