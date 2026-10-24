"""
Collect a football-injury corpus for RAG.

Sources:
  1. PMC Open Access articles via NCBI E-utilities (esearch + efetch, JATS XML)
  2. Local PDFs you download manually (FIFA 11+ manual, UEFA reports) -> data/raw/pdfs/

Output: data/processed/corpus.jsonl  (one record per section, with metadata for filtering)

Install:
  pip install biopython pymupdf
"""

import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from Bio import Entrez
from dotenv import load_dotenv
import fitz  # PyMuPDF

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------- Config ----------------
Entrez.email = os.environ.get("NCBI_EMAIL", "")    # required by NCBI; set in .env or the environment
Entrez.api_key = None                      # optional: raises limit from 3 to 10 req/s

FOOTBALL = '("soccer" OR "football")'
INJURIES = '("hamstring" OR "ACL" OR "anterior cruciate" OR "ankle sprain" OR "groin" OR "adductor" OR "meniscus" OR "calf" OR "concussion")'

# Each query is tagged with the question type the RAG must answer
QUERIES = {
    "prevention": [
        f'{FOOTBALL} AND "injury prevention"',
        '"FIFA 11+"',
        f'{FOOTBALL} AND {INJURIES} AND (prevention OR "risk factors")',
        f'{FOOTBALL} AND "Nordic hamstring"',
    ],
    "rehabilitation": [
        f'{FOOTBALL} AND {INJURIES} AND rehabilitation',
        f'{FOOTBALL} AND {INJURIES} AND (treatment OR "exercise therapy")',
    ],
    "recovery_time": [
        f'{FOOTBALL} AND "return to play"',
        f'{FOOTBALL} AND injury AND ("time loss" OR "layoff" OR "days absent")',
        '"UEFA Elite Club Injury Study" OR (Ekstrand AND football AND injury)',
    ],
}

# Sections that add noise for practical questions
SKIP_SECTIONS = ("method", "statistic", "material", "funding", "conflict",
                 "acknowledg", "author contribution", "data availability")
MAX_PER_QUERY = 200   # test run; change to 200 for the full run
BATCH = 20
ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "data" / "processed" / "corpus.jsonl"
PDF_DIR = ROOT / "data" / "raw" / "pdfs"
SLEEP = 0.4 if Entrez.api_key is None else 0.12


# ---------------- PMC ----------------
def search_pmc(query: str, retmax: int) -> list[str]:
    term = f"({query}) AND open access[filter]"
    with Entrez.esearch(db="pmc", term=term, retmax=retmax) as h:
        res = Entrez.read(h)
    time.sleep(SLEEP)
    return res["IdList"]


def text_of(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def parse_article(article: ET.Element, id_topics: dict[str, set[str]]) -> list[dict]:
    """Turn one JATS <article> into section-level records."""
    front = article.find("front/article-meta")
    if front is None:
        return []

    pmcid = ""
    for aid in front.findall("article-id"):
        if aid.get("pub-id-type") in ("pmc", "pmcid"):
            pmcid = aid.text or ""
            if not pmcid.startswith("PMC"):
                pmcid = "PMC" + pmcid
    topics = sorted(id_topics.get(pmcid.removeprefix("PMC"), set()))
    title = text_of(front.find("title-group/article-title"))
    year = text_of(front.find("pub-date/year"))
    journal = text_of(article.find("front/journal-meta/journal-title-group/journal-title"))
    license_el = front.find("permissions/license")
    license_url = ""
    if license_el is not None:
        license_url = license_el.get("{http://www.w3.org/1999/xlink}href", "") or text_of(license_el)[:200]

    base = {
        "source": "PMC",
        "topics": topics,
        "pmcid": pmcid,
        "title": title,
        "journal": journal,
        "year": year,
        "license": license_url,
        "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else "",
    }

    records = []
    abstract = text_of(front.find("abstract"))
    if abstract:
        records.append({**base, "section": "Abstract", "text": abstract})

    body = article.find("body")
    if body is not None:
        for sec in body.findall("sec"):  # top-level sections only
            sec_title = text_of(sec.find("title")) or "Untitled"
            if any(k in sec_title.lower() for k in SKIP_SECTIONS):
                continue
            paras = [text_of(p) for p in sec.iter("p")]
            text = "\n".join(p for p in paras if p)
            if len(text) > 200:
                records.append({**base, "section": sec_title, "text": text})
    return records


def fetch_pmc(id_topics: dict[str, set[str]]) -> list[dict]:
    ids = sorted(id_topics)
    records = []
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        try:
            with Entrez.efetch(db="pmc", id=",".join(chunk), retmode="xml") as h:
                root = ET.fromstring(h.read())
        except Exception as e:
            print(f"  batch {i} failed: {e}")
            time.sleep(2)
            continue
        for art in root.iter("article"):
            records.extend(parse_article(art, id_topics))
        print(f"  fetched {min(i + BATCH, len(ids))}/{len(ids)}")
        time.sleep(SLEEP)
    return records


# ---------------- Local PDFs (FIFA / UEFA) ----------------
def load_pdfs(folder: Path) -> list[dict]:
    records = []
    if not folder.exists():
        return records
    for pdf in folder.glob("*.pdf"):
        doc = fitz.open(pdf)
        for page_no, page in enumerate(doc, start=1):
            text = " ".join(page.get_text().split())
            if len(text) > 200:
                records.append({
                    "source": "FIFA" if "fifa" in pdf.name.lower() else "UEFA/other",
                    "topics": ["prevention"] if "fifa" in pdf.name.lower() else [],
                    "title": pdf.stem,
                    "section": f"page {page_no}",
                    "file": pdf.name,
                    "text": text,
                })
        print(f"  {pdf.name}: {len(doc)} pages")
    return records


# ---------------- Main ----------------
def main():
    id_topics: dict[str, set[str]] = {}
    for topic, queries in QUERIES.items():
        for q in queries:
            ids = search_pmc(q, MAX_PER_QUERY)
            print(f"[{topic}] {len(ids):>4} hits  <- {q}")
            for i in ids:
                id_topics.setdefault(i, set()).add(topic)
    print(f"Unique PMC articles: {len(id_topics)}")

    records = fetch_pmc(id_topics)
    records += load_pdfs(PDF_DIR)

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {OUT_FILE}")


if __name__ == "__main__":
    main()
