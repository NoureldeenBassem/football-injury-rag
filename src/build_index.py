"""Build the Chroma index: ingest -> chunk -> embed locally (bge-small) -> store in index/.

Run:  python -m src.build_index
Rebuilding deletes and recreates index/.
"""
import shutil

import chromadb

from src.chunk import chunk_records
from src.config import COLLECTION, EMBED_MODEL, INDEX_DIR
from src.ingest import load_records
from src.retrieve import embedder

BATCH = 256


def main():
    ids, docs, metas = chunk_records(load_records())
    print(f"Created {len(docs)} chunks")

    model = embedder()
    print(f"Embedding with {EMBED_MODEL} on {model.device}")

    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for s in range(0, len(docs), BATCH):
        e = s + BATCH
        emb = model.encode(docs[s:e], batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False)
        col.add(ids=ids[s:e], documents=docs[s:e], metadatas=metas[s:e],
                embeddings=emb.tolist())
        print(f"  {min(e, len(docs))}/{len(docs)}", flush=True)

    print(f"Done. Index saved in {INDEX_DIR.name}/ ({col.count()} chunks in '{COLLECTION}')")


if __name__ == "__main__":
    main()
