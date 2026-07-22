# BM25 keyword search -- catches exact figures/terms (GPA cutoffs, dollar
# amounts, program names) that embeddings can blur together.

from rank_bm25 import BM25Okapi


class BM25Search:
    def __init__(self, chunks):
        self.chunks = chunks
        self.corpus = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query, k=5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "metadata": self.chunks[idx]["metadata"],
                "score": float(scores[idx]),
            })
        return results
