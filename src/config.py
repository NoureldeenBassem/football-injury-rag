"""Shared paths and settings for the football-injury RAG."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "index"
EVAL_SET = ROOT / "eval" / "eval_set.csv"
OUTPUTS_DIR = ROOT / "outputs"

CORPUS = PROCESSED_DIR / "corpus_full.jsonl"   # main corpus, "topics" list per record
EXTRA_CORPUS = RAW_DIR / "cleaned_corpus.json"  # extra PMC articles, single "topic" string

# Index
COLLECTION = "football_injuries"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_SIZE = 1200
OVERLAP = 200
TOPICS = ("prevention", "rehabilitation", "recovery_time")
STATS_SOURCE = "Injury records dataset"

# Retrieval
TOP_K = 6
MIN_HITS = 3          # fewer topic-filtered hits than this -> search without the filter
MAX_STATS = 2         # recovery-time statistics records added for recovery_time questions
MAX_PER_ARTICLE = 2
FETCH_FACTOR = 8      # over-fetch so the per-article cap can still fill TOP_K

# Generation
LLM = "gemini-3.5-flash"
REQUEST_TIMEOUT_MS = 60_000


def load_api_key():
    """Make GEMINI_API_KEY available: .env file, then the environment, then (Windows)
    the user-level variable, which processes started before it was saved don't see."""
    load_dotenv(ROOT / ".env")
    if os.environ.get("GEMINI_API_KEY") or sys.platform != "win32":
        return
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            os.environ["GEMINI_API_KEY"] = winreg.QueryValueEx(k, "GEMINI_API_KEY")[0]
    except OSError:
        pass
