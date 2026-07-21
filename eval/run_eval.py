"""Evaluate the GrantMatch agent against eval/golden_profiles.json.

Ships: eligibility accuracy / citation accuracy / hallucination rate --
adapted from rag-chunking-lab's recall@k / mrr harness (eval/metrics.py) to
score the agent's final EligibilityAnswer/NeedMoreInfo instead of retrieval
alone. See eval/metrics.py for how each metric is computed.

golden_profiles.json ships 6 starter cases grounded in the real program
PDFs in data/programs/. The RUNBOOK calls for 25-30 hand-labeled profiles --
treat this file as a smoke test, not the real eval set, until it's expanded
and the expected answers are independently verified against the source docs.

Run:  python eval/run_eval.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.trace import run
from eval.metrics import aggregate, evaluate_case
from shared.loader import load_all_pdfs

CASES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_profiles.json")
PROGRAMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "programs")


def _load_source_texts():
    """{filename: full extracted text}, built straight from the PDFs -- the
    ground truth citations are checked against, independent of whatever the
    run's own retrieval happened to pull back."""
    pages = load_all_pdfs(PROGRAMS_DIR)
    texts = {}
    for p in pages:
        filename = os.path.basename(p["metadata"]["source"])
        texts[filename] = texts.get(filename, "") + "\n" + p["page_content"]
    return texts


def main():
    with open(CASES_PATH) as f:
        cases = json.load(f)
    source_texts = _load_source_texts()

    rows = []
    for case in cases:
        try:
            _, output, seq, _ = run(case["question"])
        except Exception as e:
            rows.append({"id": case["id"], "error": f"{type(e).__name__}: {e}"})
            continue

        scores = evaluate_case(output, case, source_texts)
        rows.append({
            "id": case["id"],
            "got_type": type(output).__name__,
            "seq": " -> ".join(seq) or "(none)",
            **scores,
        })

    print("\n=== Per-case ===")
    label = {True: "yes", False: "NO", None: "-"}
    for r in rows:
        if "error" in r:
            print(f"  {r['id']:<28} ERROR: {r['error']}")
            continue
        print(
            f"  {r['id']:<28} type={r['got_type']:<16} "
            f"elig={label[r['eligibility_correct']]} cite={label[r['citation_grounded']]}"
        )
        print(f"                               calls: {r['seq']}")

    scored = [r for r in rows if "error" not in r]
    agg = aggregate(scored)
    print("\n=== Aggregate ===")
    for name, value in agg.items():
        print(f"  {name:<22} {'n/a' if value != value else f'{value:.2f}'}")


if __name__ == "__main__":
    main()
