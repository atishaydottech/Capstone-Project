"""
Hybrid Search (Dense + Sparse)

Combines dense vector search (ChromaStore, semantic similarity) with BM25
sparse search (keyword matching) using Reciprocal Rank Fusion (RRF).

Why hybrid, for GrantMatch specifically: a question like "is 3.2 GPA enough
for Cal Grant A" needs dense search to match "grade point average" phrasing
variants, but the exact cutoff number (3.2, $2,000, etc.) is a literal token
BM25 is better at surfacing than an embedding is.
"""

from retrieval.bm25_search import BM25Search


def reciprocal_rank_fusion(*ranked_lists, k=60):
    """Combine multiple ranked result lists using RRF.

    Each list is a list of dicts with "text", "metadata", "score".
    Returns a single merged list sorted by RRF score.
    k=60 is the standard constant from the original RRF paper.
    """
    scores = {}
    items = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["text"][:200]
            if key not in items:
                items[key] = item
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [
        {**items[key], "score": scores[key]}
        for key in sorted_keys
    ]


class HybridSearch:
    def __init__(self, chunks, store, embed_query_fn):
        self.store = store
        self.embed_query_fn = embed_query_fn
        self.bm25 = BM25Search(chunks)

    def search(self, query, k=5, dense_k=20, sparse_k=20):
        """Retrieve using both dense and sparse, merge with RRF.

        dense_k/sparse_k: how many candidates to fetch from each before
        merging. Should be larger than final k to give RRF enough signal.
        """
        qe = self.embed_query_fn(query)
        dense_results = self.store.search(qe, k=dense_k)
        sparse_results = self.bm25.search(query, k=sparse_k)

        merged = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
        return merged[:k]
