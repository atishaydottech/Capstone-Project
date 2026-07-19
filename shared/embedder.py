"""OpenAI embeddings for RAG indexing/retrieval."""

import os
import numpy as np

MODEL = "text-embedding-3-small"

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        import openai
        _client = openai.OpenAI(api_key=api_key)
    return _client


def embed_texts(texts, batch_size=64):
    client = _get_client()
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return np.array(vectors)


def embed_query(text):
    return embed_texts([text])[0]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
