"""Load the corpus files and normalise them into one list of records with metadata.

Sources:
  data/processed/corpus_full.jsonl - main corpus (one JSON record per line, "topics" list)
  data/raw/cleaned_corpus.json     - extra PMC articles (JSON array, single "topic" string);
                                     articles already in corpus_full.jsonl are skipped.

The corpus itself is built by collect_injury_data.py and add_injury_stats.py.
"""
import json

from src.config import CORPUS, EXTRA_CORPUS

# cleaned_corpus.json uses its own topic labels; map them onto ours.
EXTRA_TOPIC_MAP = {
    "prevention": ["prevention"],
    "causes": ["prevention"],
    "risk factors": ["prevention"],
    "epidemiology": ["prevention"],
    "treatment": ["rehabilitation"],
    "recovery": ["rehabilitation", "recovery_time"],
    "return to play": ["rehabilitation", "recovery_time"],
}


def load_records(verbose=True):
    records, seen_pmc = [], set()
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records.append(r)
                if r.get("pmcid"):
                    seen_pmc.add(r["pmcid"].upper())
    if verbose:
        print(f"Loaded {len(records)} records from {CORPUS.name}")

    if EXTRA_CORPUS.exists():
        with open(EXTRA_CORPUS, encoding="utf-8") as f:
            extra = json.load(f)
        added = 0
        for r in extra:
            pmcid = "PMC" + str(r["id"])
            if pmcid.upper() in seen_pmc:
                continue
            seen_pmc.add(pmcid.upper())
            records.append({
                "source": "PMC",
                "pmcid": pmcid,
                "title": r.get("title", ""),
                "section": r.get("topic", ""),
                "injury": r.get("injury", ""),
                "url": r.get("url", ""),
                "topics": EXTRA_TOPIC_MAP.get(r.get("topic", "").lower(), []),
                "text": r.get("text", ""),
            })
            added += 1
        if verbose:
            print(f"Added {added} new records from {EXTRA_CORPUS.name} "
                  f"({len(extra) - added} duplicates skipped)")
    return records


if __name__ == "__main__":
    load_records()
