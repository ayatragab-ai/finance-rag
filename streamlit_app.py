# ============================================================
# streamlit_app.py
# Financial QA RAG Assistant (FinanceQA / 10-K filings).
#
# Flow:  type a financial question  ->  the app retrieves the most
# relevant filing passages (hybrid BM25 + embeddings), then answers
# with an LLM via OpenRouter, grounded and with cited sources.
# ============================================================

from importlib import import_module

import streamlit as st

vectors = import_module("04_vector_representation")
rag = import_module("07_prompting")

st.set_page_config(page_title="Financial QA RAG", page_icon="📊", layout="centered")

# --- OpenRouter credentials from Streamlit secrets (if present) ---
try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass


# --- Build the search index once (loads FinanceQA + builds embeddings) ---
@st.cache_resource(show_spinner="Loading financial data and building the index (first run only)…")
def get_index():
    return vectors.build_index()


st.title("📊 Financial QA RAG Assistant")
st.caption(
    "Ask a question about the company 10-K filings loaded with this app. "
    "Answers are grounded only in the retrieved passages and cite their sources."
)

# --- Load the index (with a friendly error if the dataset can't load) ---
try:
    index = get_index()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the local data file. Details: {exc}")
    st.stop()

# --- Sidebar: project info ---
with st.sidebar:
    st.header("About")
    st.markdown(
        "This app answers questions over company **10-K filings** "
        "using a Retrieval-Augmented Generation (RAG) pipeline."
    )
    st.markdown("**Data:** local financial filings (10-K)")
    st.markdown(f"**Documents loaded:** {len(index.documents)}")
    st.markdown(f"**Chunks indexed:** {len(index.chunks)}")
    retrieval_mode = "Hybrid (BM25 + embeddings)" if index.semantic_ready else "BM25 only"
    st.markdown(f"**Retrieval:** {retrieval_mode}")
    st.markdown(f"**LLM (OpenRouter):** `{rag.OPENROUTER_MODEL}`")
    if not rag.OPENROUTER_API_KEY:
        st.warning("No OpenRouter key found. Add it in Settings → Secrets.")

# --- Keep the current question in session state so example buttons can fill it ---
if "question" not in st.session_state:
    st.session_state.question = ""

# --- Clickable example questions pulled from the loaded data ---
examples = [d["question"] for d in index.documents if d.get("question")][:4]
if examples:
    st.markdown("**Try an example:**")
    cols = st.columns(2)
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, key=f"ex_{i}"):
            st.session_state.question = example

# --- Question input ---
question = st.text_area(
    "Your financial question",
    key="question",
    placeholder="e.g. What was the percentage change in worldwide sales from 2022 to 2023?",
    height=110,
)

# --- Answer ---
ask = st.button("Get answer", type="primary")
if ask and question.strip():
    with st.spinner("Searching the filings and generating an answer…"):
        answer, sources, grounded = rag.answer_question(index, question)

    st.subheader("Answer")

    if grounded:
        st.success("✅ Grounded answer — based on the stored financial filings.")
    else:
        st.warning(
            "⚠️ Not found in the stored filings. This answer comes from the "
            "model's general knowledge, so verify it before relying on it."
        )

    st.write(answer)

    if grounded:
        with st.expander(f"Sources used ({len(sources)})", expanded=True):
            for i, source in enumerate(sources, start=1):
                st.markdown(
                    f"**[Source {i}] {source['doc_title']}**  ·  score: {source['score']:.3f}"
                )
                st.write(source["chunk_text"])
elif ask:
    st.warning("Please type a question first.")
