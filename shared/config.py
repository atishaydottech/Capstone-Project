# ingest.py writes here, agent/tools.py reads from here -- keep them in sync.

import os

COLLECTION_NAME = "grantmatch"
PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_data")
