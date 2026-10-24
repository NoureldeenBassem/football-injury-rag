"""Streamlit chat UI for the football-injury RAG.

Run from the project root:  streamlit run app/ui.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

import httpx
import streamlit as st
from google.genai import errors

from src.config import LLM
from src.generate import DailyQuotaExceeded, answer_stream, rewrite_query
from src.retrieve import collection, retrieve

st.set_page_config(page_title="Football Injury Assistant", page_icon=":material/sports_soccer:")

SUGGESTIONS = {
    ":blue[:material/shield:] Prevent a hamstring injury": "How can I prevent a hamstring injury?",
    ":green[:material/healing:] Ankle sprain (Franco-Arab)": "ezay at3afa men eltewa2 el kahl?",
    ":orange[:material/timer:] ACL recovery time": "How long does an ACL tear take to recover from?",
}
TOPIC_LABELS = {"prevention": "Prevention", "rehabilitation": "Rehabilitation",
                "recovery_time": "Recovery time", None: "No topic"}


@st.cache_resource(show_spinner="Loading the index and embedding model...")
def load_index():
    return collection().count()


def render_sources(hits):
    with st.expander(f"Sources ({len(hits)})", icon=":material/menu_book:"):
        for i, h in enumerate(hits, 1):
            m = h["meta"]
            title = f"[{m['title']}]({m['url']})" if m.get("url") else m["title"]
            extra = " · ".join(x for x in (m["source"], m.get("year")) if x)
            st.markdown(f"**[{i}]** {title}  \n:gray[{extra}]")


def render_search(info):
    st.caption(f":material/search: {info['query']} · {TOPIC_LABELS[info['topic']]}"
               + ("" if info["filtered"] else " (no topic filter)")
               + (f" · stats: {', '.join(info['injuries'])}" if info["injuries"] else ""))


chunks = load_index()

with st.sidebar:
    st.title(":material/sports_soccer: Football Injury Assistant")
    st.write("Answers about **prevention**, **rehabilitation** and **recovery time**, "
             "with citations from PMC papers, FIFA 11+ manuals and recovery statistics.")
    st.write("Ask in English, Arabic or Franco-Arab.")
    st.caption(f"{chunks:,} chunks indexed · {LLM}")
    if st.button("New chat", icon=":material/refresh:", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.caption(":material/info: Not medical advice. See a doctor or physiotherapist "
               "for a personal diagnosis.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("search"):
            render_search(msg["search"])
        st.markdown(msg["content"])
        if msg.get("hits"):
            render_sources(msg["hits"])

prompt = st.chat_input("Ask about a football injury...", submit_mode="disable")

if not st.session_state.messages and not prompt:
    welcome = st.empty()
    with welcome.container():
        st.markdown("### How can I help?")
        picked = st.pills("Try asking:", list(SUGGESTIONS), label_visibility="collapsed")
    if picked:
        prompt = SUGGESTIONS[picked]
        welcome.empty()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.status("Searching the corpus...", expanded=False) as status:
                rq = rewrite_query(prompt, on_retry=lambda m: status.update(
                    label=f"Searching the corpus... (Gemini is busy: {m})"))
                hits, filtered = retrieve(rq["query"], rq["topic"], injuries=rq["injuries"])
                status.update(label=f"Found {len(hits)} sources", state="complete")
            search = {"query": rq["query"], "topic": rq["topic"], "filtered": filtered,
                      "injuries": rq["injuries"]}
            render_search(search)
            notice = st.empty()
            text = st.write_stream(answer_stream(
                prompt, hits, rq.get("language", "the user's language"),
                on_retry=lambda m: notice.warning(f"Gemini is busy: {m}", icon=":material/hourglass:")))
            notice.empty()
            render_sources(hits)
            st.session_state.messages.append(
                {"role": "assistant", "content": text, "hits": hits, "search": search})
        except DailyQuotaExceeded as e:
            st.error(str(e), icon=":material/block:")
            st.session_state.messages.pop()
        except errors.APIError as e:
            st.error(f"Gemini error {e.code}: {e.message}", icon=":material/error:")
            st.session_state.messages.pop()
        except httpx.HTTPError as e:
            st.error(f"Could not reach Gemini: {e}", icon=":material/wifi_off:")
            st.session_state.messages.pop()
