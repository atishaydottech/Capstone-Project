"""Shared constants for wherever the persisted Chroma collection lives.

Single source of truth for ingest.py (writer) and agent/tools.py (reader) so
they can't silently drift onto different collections/paths.
"""

import os

COLLECTION_NAME = "grantmatch"
PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_data")
