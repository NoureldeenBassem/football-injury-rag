"""Split records into overlapping character chunks with filterable metadata."""
from src.config import CHUNK_SIZE, OVERLAP, TOPICS


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into ~size-char chunks with overlap, preferring whitespace breaks."""
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def chunk_records(records):
    """Return (ids, documents, metadatas) for all records; topics become boolean flags."""
    ids, docs, metas = [], [], []
    for i, r in enumerate(records):
        topics = set(r.get("topics") or [])
        for j, chunk in enumerate(chunk_text(r.get("text", ""))):
            ids.append(f"r{i}_c{j}")
            docs.append(chunk)
            meta = {
                "source": r.get("source", ""),
                "title": r.get("title", ""),
                "section": r.get("section", ""),
                "url": r.get("url", "") or "",
                "year": str(r.get("year", "") or ""),
                "injury": r.get("injury", "") or "",
            }
            for t in TOPICS:
                meta[f"topic_{t}"] = t in topics
            metas.append(meta)
    return ids, docs, metas
