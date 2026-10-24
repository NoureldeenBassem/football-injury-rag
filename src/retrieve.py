"""Vector search over the Chroma index, with topic filtering and statistics lookup."""
import chromadb
import torch
from sentence_transformers import SentenceTransformer

from src.config import (COLLECTION, EMBED_MODEL, FETCH_FACTOR, INDEX_DIR, MAX_PER_ARTICLE,
                        MIN_HITS, QUERY_PREFIX, STATS_SOURCE, TOP_K)

_model = None
_col = None
_stat_injuries = None


def embedder():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(EMBED_MODEL, device=device)
    return _model


def collection():
    global _col
    if _col is None:
        embedder()
        _col = chromadb.PersistentClient(path=str(INDEX_DIR)).get_collection(COLLECTION)
    return _col


def stat_injuries():
    """Injury names that have a recovery-time statistics record."""
    global _stat_injuries
    if _stat_injuries is None:
        res = collection().get(where={"source": STATS_SOURCE}, include=["metadatas"])
        _stat_injuries = sorted({m["injury"] for m in res["metadatas"] if m.get("injury")})
    return _stat_injuries


def stats_hits(injuries):
    """Recovery-time statistics records for the given injury names, in that order."""
    if not injuries:
        return []
    res = collection().get(where={"$and": [{"source": STATS_SOURCE},
                                           {"injury": {"$in": list(injuries)}}]},
                           include=["documents", "metadatas"])
    by_name = {m["injury"]: {"text": d, "meta": m, "distance": None}
               for d, m in zip(res["documents"], res["metadatas"])}
    return [by_name[n] for n in injuries if n in by_name]


def retrieve(query, topic=None, k=TOP_K, injuries=None):
    """Top-k chunks, filtered by topic; falls back to no filter if too few hits.

    Statistics records for `injuries` (recovery-time questions) go first. At most
    MAX_PER_ARTICLE chunks come from the same article, so answers cite more sources.
    Returns (hits, filtered).
    """
    col = collection()
    stats = stats_hits(injuries)
    k_search = k - len(stats)
    emb = embedder().encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()

    def search(where):
        res = col.query(query_embeddings=emb, n_results=k_search * FETCH_FACTOR,
                        where=where, include=["documents", "metadatas", "distances"])
        seen = {h["text"] for h in stats}
        per_article, hits = {}, []
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            key = m.get("url") or m.get("title")
            if d in seen or per_article.get(key, 0) >= MAX_PER_ARTICLE:
                continue
            seen.add(d)
            per_article[key] = per_article.get(key, 0) + 1
            hits.append({"text": d, "meta": m, "distance": dist})
            if len(hits) == k_search:
                break
        return hits

    hits = search({f"topic_{topic}": True}) if topic else []
    filtered = bool(topic) and len(hits) >= MIN_HITS
    if not filtered:
        hits = search(None)
    return stats + hits, filtered
