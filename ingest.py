"""Offline ingestion: load -> section_wise chunk -> OpenAI embed -> ChromaDB.add.

Builds the persisted collection that agent/tools.py's retriever_tool reads
from at query time. Run this once (and again whenever data/programs/ changes):

    python ingest.py
"""

import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from shared.loader import load_all_pdfs
from shared.embedder import embed_texts
from chunking import section_wise
from vectordb.chroma_store import ChromaStore
from shared.config import COLLECTION_NAME, PERSIST_DIR

PROGRAMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "programs")


def main():
    pages = load_all_pdfs(PROGRAMS_DIR)
    print(f"Loaded {len(pages)} pages from {PROGRAMS_DIR}")

    chunks = section_wise.chunk(pages, chunk_size=800, chunk_overlap=80)
    print(f"Chunked into {len(chunks)} sections")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    print(f"Embedded {len(embeddings)} chunks ({embeddings.shape[1]}d)")

    store = ChromaStore(collection_name=COLLECTION_NAME, persist_dir=PERSIST_DIR, reset=True)
    store.add(chunks, embeddings)
    print(f"Indexed {store.count} chunks into '{COLLECTION_NAME}' at {PERSIST_DIR}")


if __name__ == "__main__":
    main()
