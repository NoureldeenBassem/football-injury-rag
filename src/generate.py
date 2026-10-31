"""Query rewriting, Gemini answer generation with [n] citations, and the rag() pipeline.

Run:
  python -m src.generate            # interactive Q&A in the terminal
  python -m src.generate --models   # list available Gemini models
"""
import json
import re
import sys
import time

import httpx
from google import genai
from google.genai import errors, types

from src.config import LLMS, MAX_STATS, REQUEST_TIMEOUT_MS, TOPICS, load_api_key
from src.retrieve import collection, retrieve, stat_injuries

_client = None


def client():
    global _client
    if _client is None:
        load_api_key()
        # The timeout (ms) stops a stalled request from hanging forever; with_retry
        # then tries again.
        _client = genai.Client(http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
    return _client


RETRY_CODES = {429: "rate limited", 500: "server error", 503: "model overloaded",
               504: "timeout"}


class DailyQuotaExceeded(Exception):
    """Every model in LLMS has used up its free-tier daily requests."""


# model -> time (epoch seconds) until which its daily quota is used up. Shared by all
# sessions of the app process, so one user hitting the limit moves everyone on.
_exhausted = {}


def is_daily_quota(e):
    return isinstance(e, errors.APIError) and e.code == 429 and "PerDay" in str(e)


def mark_exhausted(model, e):
    """Skip `model` until Gemini says its quota resets (retryDelay), default 1 hour."""
    m = re.search(r"'retryDelay': '(\d+)s'", str(e))
    _exhausted[model] = time.time() + (int(m.group(1)) if m else 3600)
    print(f"[{model}: daily limit used up, switching model]", file=sys.stderr, flush=True)


def available_models():
    now = time.time()
    models = [m for m in LLMS if _exhausted.get(m, 0) <= now]
    if not models:
        raise DailyQuotaExceeded(
            "All Gemini models in the free tier have used up their daily requests "
            f"({', '.join(LLMS)}). They reset within 24 hours.")
    return models


def current_model():
    """The model that will be tried first for the next request."""
    return available_models()[0]


def retry_reason(e):
    """Why error e is worth retrying with the same model, or None if it isn't."""
    if isinstance(e, errors.APIError):
        return f"{RETRY_CODES[e.code]} ({e.code})" if e.code in RETRY_CODES else None
    if isinstance(e, httpx.TimeoutException):
        return "request timed out"
    if isinstance(e, httpx.TransportError):
        return "connection error"
    return None


def wait_before_retry(attempt, reason, on_retry=None):
    wait = 20 * (attempt + 1)
    msg = f"{reason}, retrying in {wait}s"
    if on_retry:
        on_retry(msg)
    else:
        print(f"\n[{msg}]", file=sys.stderr, flush=True)
    time.sleep(wait)


def skip_model(model, e):
    """Move on to the next model if `model` is used up for today or not available."""
    if is_daily_quota(e):
        mark_exhausted(model, e)
        return True
    if isinstance(e, errors.APIError) and e.code == 404:  # model retired / not offered
        _exhausted[model] = time.time() + 24 * 3600
        print(f"[{model}: not available, switching model]", file=sys.stderr, flush=True)
        return True
    return False


def with_retry(fn, tries=5, on_retry=None):
    """Call fn(model) with the first available model. A used-up model is swapped for the
    next one in LLMS; a rate limit or temporary error waits (growing delay) and retries."""
    attempt = 0
    while True:
        model = current_model()  # raises DailyQuotaExceeded once every model is used up
        try:
            return fn(model)
        except (errors.APIError, httpx.HTTPError) as e:
            if skip_model(model, e):
                continue
            reason = retry_reason(e)
            if reason is None or attempt == tries - 1:
                raise
            wait_before_retry(attempt, reason, on_retry)
            attempt += 1


REWRITE_PROMPT = """You turn football (soccer) injury questions into a search query.
The question may be in English, Arabic, or Franco-Arab (Egyptian Arabic written in
Latin letters and digits, e.g. "ezay" = how, "at3afa" = recover, "eltewa2" = sprain,
"kahl" = ankle, "rokba" = knee).

Return JSON with:
  "query": a concise English search query capturing the injury and what is asked;
           spell out abbreviations (e.g. ACL -> anterior cruciate ligament (ACL))
  "topic": one of "prevention" (avoiding injury), "rehabilitation" (treatment,
           exercises, how to recover), "recovery_time" (how long until return),
           or "none" if none fits
  "language": the language the user wrote in (e.g. "English", "Arabic",
           "Franco-Arab")
  "injuries": only if topic is "recovery_time": up to {max_stats} names from the
           allowed list that best match the injury asked about (most specific
           first, e.g. an ACL tear -> "Cruciate ligament tear"); otherwise []

Allowed injury names: {injury_names}

Question: {question}"""


def rewrite_query(question, on_retry=None):
    """Turn a question (any language) into {query, topic, language, injuries}."""
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(type=types.Type.STRING),
            "topic": types.Schema(type=types.Type.STRING,
                                  enum=list(TOPICS) + ["none"]),
            "language": types.Schema(type=types.Type.STRING),
            "injuries": types.Schema(type=types.Type.ARRAY,
                                     items=types.Schema(type=types.Type.STRING)),
        },
        required=["query", "topic", "language", "injuries"],
    )
    resp = with_retry(lambda model: client().models.generate_content(
        model=model,
        contents=REWRITE_PROMPT.format(question=question, max_stats=MAX_STATS,
                                       injury_names="; ".join(stat_injuries())),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0,
        ),
    ), on_retry=on_retry)
    data = json.loads(resp.text)
    if data.get("topic") not in TOPICS:
        data["topic"] = None
    known = {n.lower(): n for n in stat_injuries()}
    picked = [known[n.strip().lower()] for n in data.get("injuries", []) if n.strip().lower() in known]
    data["injuries"] = picked[:MAX_STATS] if data["topic"] == "recovery_time" else []
    return data


TRANSCRIBE_PROMPT = """Transcribe this voice message, a question about a football (soccer)
injury, word for word. The speaker may use English, Arabic or Egyptian Arabic; write
Arabic speech in Arabic script. Return only the transcript, with no notes. If there is
no speech, return nothing."""


def transcribe(audio, mime_type="audio/wav", on_retry=None):
    """Turn a recorded voice question (audio bytes) into text."""
    resp = with_retry(lambda model: client().models.generate_content(
        model=model,
        contents=[TRANSCRIBE_PROMPT, types.Part.from_bytes(data=audio, mime_type=mime_type)],
        config=types.GenerateContentConfig(temperature=0),
    ), on_retry=on_retry)
    return (resp.text or "").strip()


ANSWER_PROMPT = """You are a football (soccer) injury assistant. Answer the user's question
using ONLY the numbered sources below. Cite sources inline as [n]. If the sources do
not contain the answer, say so. Keep it practical and well structured. Remind the user
to consult a doctor or physiotherapist for a personal diagnosis.

Answer in the same language and style the user wrote in ({language}). If they wrote
Franco-Arab, answer in Franco-Arab.

Sources:
{sources}

Question: {question}"""


def format_sources(hits):
    return "\n\n".join(
        f"[{i}] {h['meta']['title']} ({h['meta']['source']}, {h['meta']['section']})\n{h['text']}"
        for i, h in enumerate(hits, 1))


def source_list(hits):
    lines = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        extra = ", ".join(x for x in (m.get("year"), m.get("url")) if x)
        lines.append(f"[{i}] {m['title']} - {m['source']}" + (f" ({extra})" if extra else ""))
    return "\n".join(lines)


def answer_stream(question, hits, language="the user's language", tries=5, on_retry=None,
                  info=None):
    """Yield the answer as text pieces. Before the first piece, a used-up model is swapped
    for the next one and temporary errors are retried. The model used goes in info["model"]."""
    prompt = ANSWER_PROMPT.format(language=language, sources=format_sources(hits),
                                  question=question)
    attempt = 0
    while True:
        model = current_model()
        started = False
        try:
            for chunk in client().models.generate_content_stream(model=model, contents=prompt):
                if chunk.text:
                    if not started and info is not None:
                        info["model"] = model
                    started = True
                    yield chunk.text
            return
        except (errors.APIError, httpx.HTTPError) as e:
            if not started and skip_model(model, e):
                continue
            reason = retry_reason(e)
            if started or reason is None or attempt == tries - 1:
                raise
            wait_before_retry(attempt, reason, on_retry)
            attempt += 1


def answer(question, hits, language="the user's language", echo=False, info=None):
    """Full answer text; with echo=True it is also printed while it streams."""
    parts = []
    for text in answer_stream(question, hits, language, info=info):
        if echo:
            print(text, end="", flush=True)
        parts.append(text)
    if echo:
        print()
    return "".join(parts)


def rag(question, echo=False):
    """Rewrite -> retrieve -> answer. Returns rewrite, filter flag, hits, answer and the
    model that wrote the answer."""
    rq = rewrite_query(question)
    hits, filtered = retrieve(rq["query"], rq["topic"], injuries=rq["injuries"])
    if echo:
        print(search_line(rq, filtered) + "\n")
    info = {}
    text = answer(question, hits, rq.get("language", "the user's language"), echo=echo, info=info)
    if echo:
        print(f"\n(answered by {info.get('model')})")
    return {"rewrite": rq, "filtered": filtered, "hits": hits, "answer": text,
            "model": info.get("model")}


def search_line(rq, filtered):
    return (f"(search: {rq['query']!r} | topic: {rq['topic'] or 'none'}"
            f"{'' if filtered else ', no filter'}"
            f"{' | stats: ' + ', '.join(rq['injuries']) if rq['injuries'] else ''})")


def list_models():
    for m in client().models.list():
        if "generateContent" in (m.supported_actions or []):
            print(m.name)


def main():
    if "--models" in sys.argv:
        list_models()
        return
    collection()
    print("Football injury assistant. Ask in English, Arabic or Franco-Arab (empty line to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        try:
            result = rag(q, echo=True)
            print("\nSources:\n" + source_list(result["hits"]))
        except DailyQuotaExceeded as e:
            print(e)
            break
        except (errors.APIError, httpx.HTTPError) as e:
            print(f"Gemini error: {e}")


if __name__ == "__main__":
    main()
