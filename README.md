# Football Injury RAG

A retrieval-augmented Q&A assistant for football (soccer) injuries: **prevention**,
**rehabilitation** and **recovery time**. Answers cite their sources ([n]) and work in
English, Arabic and Franco-Arab (Egyptian Arabic in Latin letters).

- **Corpus:** 5,723 records (PMC open-access papers, FIFA 11+ manuals, recovery-time
  statistics from professional-player injury records) + 94 extra PMC articles
- **Index:** 35,597 chunks (~1,200 chars, 200 overlap) in ChromaDB, embedded locally with
  `BAAI/bge-small-en-v1.5`
- **LLM:** Gemini free tier via the `google-genai` SDK. `LLMS` in `src/config.py` lists
  several models; when one hits its daily limit the app switches to the next.

## Project structure

```
sport-rag/
├── data/
│   ├── raw/            # provided inputs: PDFs, injury CSVs, cleaned_corpus.json
│   └── processed/      # corpus.jsonl, injury_stats.csv, corpus_full.jsonl (final corpus)
├── index/              # persisted Chroma index (gitignored; rebuild with src.build_index)
├── src/
│   ├── config.py       # paths, model names, retrieval settings
│   ├── collect_injury_data.py  # fetch PMC articles + parse PDFs -> corpus.jsonl
│   ├── add_injury_stats.py     # injury CSVs -> recovery-time records -> corpus_full.jsonl
│   ├── ingest.py       # load corpus files, dedupe, normalise metadata
│   ├── chunk.py        # chunking with topic flags
│   ├── build_index.py  # ingest -> chunk -> embed -> Chroma
│   ├── retrieve.py     # vector search, topic filter, stats lookup, per-article cap
│   ├── generate.py     # query rewrite, prompt, Gemini call, citations, rag()
│   └── evaluate.py     # run eval/eval_set.csv, save results to outputs/
├── app/ui.py           # Streamlit chat interface
├── eval/eval_set.csv   # test questions with expected topic
├── outputs/            # baseline answers, eval results
├── requirements.txt        # app + pipeline (CPU torch on Linux)
├── requirements-data.txt   # + data collection packages
├── .env.example        # never commit real keys
└── README.md
```

## Setup

```bash
pip install -r requirements.txt        # app + RAG pipeline
pip install -r requirements-data.txt   # also the data collection scripts
```

For GPU embeddings, install the CUDA build of torch (see `requirements.txt`).

The repo includes `data/processed/` (the final corpus) but not `data/raw/` (FIFA PDFs and
third-party injury datasets) or `index/`. On first start the app downloads the prebuilt
index (245 MB, [release `index-v1`](https://github.com/NoureldeenBassem/football-injury-rag/releases/tag/index-v1))
into `index/`. To build it yourself instead, run `python -m src.build_index`; without
`data/raw/cleaned_corpus.json` it has slightly fewer chunks than listed above.

Set your Gemini API key, either as an environment variable `GEMINI_API_KEY` or in a
`.env` file (copy `.env.example`).

## Usage

All commands run from the project root.

```bash
python -m src.build_index         # build index/ (~15 min on GPU, longer on CPU)
streamlit run app/ui.py           # chat UI at http://localhost:8501
python -m src.generate            # chat in the terminal
python -m src.generate --models   # list available Gemini models
python -m src.evaluate            # run the eval set -> outputs/
```

## Deploy on Streamlit Community Cloud

- **Main file path:** `app/ui.py`
- **Advanced settings → Python version:** 3.12
- **Advanced settings → Secrets:** `GEMINI_API_KEY = "your-key"`

The first start downloads the index and the embedding model, so it takes a few minutes.
Everyone using the deployed app shares your Gemini quota.

## How it works

1. **Rewrite:** Gemini turns the question (any language) into an English search query, a
   topic (`prevention` / `rehabilitation` / `recovery_time`) and, for recovery-time
   questions, the matching injury names from the statistics records.
2. **Retrieve:** top 6 chunks filtered by topic (no filter if fewer than 3 hits), at most
   2 chunks per article. For recovery-time questions the matching statistics records come
   first. These records say "cruciate ligament", never "ACL", so they are matched by injury
   name rather than by vector search.
3. **Generate:** Gemini answers from the numbered sources only, with [n] citations, in the
   user's language.

Gemini calls retry on rate limits (429), overload (503) and timeouts. The free tier allows
about 20 requests per model per day, and each question uses 2. When a model's daily limit
is used up (or the model is unavailable), the next model in `LLMS` takes over until the
limit resets, so the app stays free and works for several times longer each day. The
answer shows which model wrote it. Lite models answer when the main ones are used up and
are somewhat less precise.

*Not medical advice: see a doctor or physiotherapist for a personal diagnosis.*
