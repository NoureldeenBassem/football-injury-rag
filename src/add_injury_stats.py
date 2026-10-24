"""
Turn tabular injury records into RAG-friendly text, then merge with corpus.jsonl.

Inputs:
  data/raw/player_injuries.csv          (player_id, season_name, injury_reason, days_missed, games_missed, ...)
  data/raw/full_dataset_thesis_-_1.csv  (Injury, Days, Games missed, player_position, league, ...)
  data/processed/corpus.jsonl           (output of collect_injury_data.py)

Outputs:
  data/processed/injury_stats.csv       (one row per injury type, for checking)
  data/processed/corpus_full.jsonl      (corpus.jsonl + recovery-time records)  <- use this for the RAG

Safe to run more than once: corpus_full.jsonl is rebuilt every time.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

MIN_CASES = 30          # skip injury types with too few cases for reliable stats
MAX_DAYS = 730          # drop records longer than 2 years (data errors / careers ended)

# Absences that are not sports injuries
NOT_INJURY = {
    "unknown injury", "ill", "illness", "corona virus", "covid-19", "quarantine",
    "flu", "influenza", "cold", "fever", "fitness", "rest", "suspension",
    "lack of fitness", "personal reasons", "family reasons", "virus",
    "infection", "gastric problems", "stomach problems", "food poisoning",
}


def load_injuries() -> pd.DataFrame:
    a = pd.read_csv(RAW / "player_injuries.csv")
    a = pd.DataFrame({
        "injury": a["injury_reason"],
        "days": pd.to_numeric(a["days_missed"], errors="coerce"),
        "games": pd.to_numeric(a["games_missed"], errors="coerce"),
        "position": None,
    })

    thesis = sorted(RAW.glob("full_dataset_thesis*.csv"))
    if not thesis:
        raise FileNotFoundError(f"No file starting with 'full_dataset_thesis' in {RAW}")
    b = pd.read_csv(thesis[0])
    b = pd.DataFrame({
        "injury": b["Injury"],
        "days": pd.to_numeric(b["Days"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"),
        "games": pd.to_numeric(b["Games missed"], errors="coerce"),
        "position": b["player_position"],
    })

    df = pd.concat([a, b], ignore_index=True)
    df["injury"] = df["injury"].astype(str).str.strip()
    df["key"] = df["injury"].str.lower()
    df = df[~df["key"].isin(NOT_INJURY)]
    df = df[df["days"].between(1, MAX_DAYS)]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("key")
    stats = pd.DataFrame({
        "injury": g["injury"].agg(lambda s: s.mode().iat[0]),   # most common spelling
        "cases": g.size(),
        "median_days": g["days"].median(),
        "p25_days": g["days"].quantile(0.25),
        "p75_days": g["days"].quantile(0.75),
        "p90_days": g["days"].quantile(0.90),
        "median_games": g["games"].median(),
    })
    stats = stats[stats["cases"] >= MIN_CASES].sort_values("cases", ascending=False)
    return stats.round(0).astype({"cases": int})


def position_note(df: pd.DataFrame, key: str) -> str:
    sub = df[(df["key"] == key) & df["position"].notna()]
    if len(sub) < 60:
        return ""
    by_pos = sub.groupby("position")["days"].agg(["median", "size"])
    by_pos = by_pos[by_pos["size"] >= 15].sort_values("median")
    if len(by_pos) < 2:
        return ""
    lo, hi = by_pos.iloc[0], by_pos.iloc[-1]
    return (f" By position, {by_pos.index[-1]} players had the longest median absence "
            f"({int(hi['median'])} days) and {by_pos.index[0]} players the shortest "
            f"({int(lo['median'])} days).")


def to_records(stats: pd.DataFrame, df: pd.DataFrame) -> list[dict]:
    records = []
    for key, r in stats.iterrows():
        text = (
            f"How long does {'an' if r['injury'][0].lower() in 'aeiou' else 'a'} {r['injury'].lower()} take to recover from in professional football? "
            f"Based on {r['cases']:,} recorded cases of professional players, the median time out "
            f"was {int(r['median_days'])} days. Half of the players returned between "
            f"{int(r['p25_days'])} and {int(r['p75_days'])} days, and 90% returned within "
            f"{int(r['p90_days'])} days. The median number of matches missed was "
            f"{int(r['median_games'])}."
            + position_note(df, key)
            + " Recovery depends on severity, so a specific case can fall outside this range."
        )
        records.append({
            "source": "Injury records dataset",
            "topics": ["recovery_time"],
            "title": f"Recovery time statistics: {r['injury']}",
            "section": "Statistics",
            "injury": r["injury"],
            "text": text,
        })
    return records


def main():
    df = load_injuries()
    print(f"Injury records after cleaning: {len(df):,}")

    stats = summarize(df)
    stats.to_csv(PROCESSED / "injury_stats.csv")
    print(f"Injury types with >= {MIN_CASES} cases: {len(stats)}  -> injury_stats.csv")

    records = to_records(stats, df)

    out = PROCESSED / "corpus_full.jsonl"
    n_corpus = 0
    with out.open("w", encoding="utf-8") as f:
        corpus = PROCESSED / "corpus.jsonl"
        if corpus.exists():
            with corpus.open(encoding="utf-8") as src:
                for line in src:
                    f.write(line if line.endswith("\n") else line + "\n")
                    n_corpus += 1
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {n_corpus + len(records):,} records to {out} "
          f"({n_corpus:,} from corpus.jsonl + {len(records)} recovery-time records)")


if __name__ == "__main__":
    main()
