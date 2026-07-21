"""
GrantMatch Eval Metrics

Adapted from rag-chunking-lab/eval/metrics.py's recall@k / mrr pattern, but
scores the agent's final EligibilityAnswer/NeedMoreInfo against a labeled
student profile instead of scoring retrieval alone.

eligibility_correct: did the agent return the expected output type, and (for
                      EligibilityAnswer) the expected yes/no/partial verdict?

citation_grounded:    is cited_clause actually present in the real source
                      document text for cited_source? This checks the clause
                      against the real PDF text, not just a keyword match --
                      RAG_LESSONS.md #7's fragile-substring-eval lesson
                      applies here too: a keyword can appear in an unrelated
                      chunk and look like a "correct" citation when it isn't.

hallucination_rate:   1 - citation_accuracy. An EligibilityAnswer whose
                      citation isn't grounded in a real document is treated
                      as a hallucinated citation, regardless of whether the
                      yes/no/partial verdict happened to be right.
"""

import re


def _normalize(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def eligibility_correct(output, case):
    """None when the output TYPE doesn't match what the case expects -- that's
    a type-level miss (e.g. agent asked for more info instead of answering),
    scored separately from whether the verdict itself was right."""
    got_type = type(output).__name__
    if got_type != case["expected_output"]:
        return None
    if case["expected_output"] != "EligibilityAnswer":
        return True  # correctly returned NeedMoreInfo; nothing else to check
    return output.eligible == case["expected_eligible"]


def citation_grounded(output, source_texts, min_run=25):
    """Is cited_clause a real, contiguous run of text from cited_source's actual
    document? None when the case didn't produce an EligibilityAnswer at all.

    source_texts: {filename: full_extracted_text} for every program PDF, built
    independently of whatever the run actually retrieved -- this is checking
    the answer against the ground-truth corpus, not against the agent's own
    retrieval results (which could themselves be wrong).
    """
    if type(output).__name__ != "EligibilityAnswer":
        return None

    clause = _normalize(output.cited_clause)
    if not clause:
        return False

    # Loosely match cited_source (free text, e.g. "Cal Grant A" or the filename)
    # to a real program file by substring overlap either direction.
    cited = _normalize(output.cited_source)
    matched_texts = [
        text for filename, text in source_texts.items()
        if _normalize(filename).replace(".pdf", "").replace("_", " ") in cited
        or cited in _normalize(filename)
    ]
    # Citation didn't name a recognizable real source at all -> not grounded.
    # (Deliberately not falling back to searching every file: a citation that
    # can't be tied to a real, named source is itself a grounding failure.)
    if not matched_texts:
        return False

    window = clause[:min_run] if len(clause) >= min_run else clause
    return any(window in text for text in (_normalize(t) for t in matched_texts))


def evaluate_case(output, case, source_texts):
    return {
        "eligibility_correct": eligibility_correct(output, case),
        "citation_grounded": citation_grounded(output, source_texts),
    }


def aggregate(results):
    elig = [r["eligibility_correct"] for r in results if r["eligibility_correct"] is not None]
    cite = [r["citation_grounded"] for r in results if r["citation_grounded"] is not None]
    citation_accuracy = (sum(cite) / len(cite)) if cite else float("nan")
    return {
        "eligibility_accuracy": (sum(elig) / len(elig)) if elig else float("nan"),
        "citation_accuracy": citation_accuracy,
        "hallucination_rate": (1 - citation_accuracy) if cite else float("nan"),
    }
