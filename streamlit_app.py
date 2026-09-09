from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from ragchunk import AdaptiveChunkingPipeline
from ragchunk.embeddings import build_gemini_embed_fn
from ragchunk.generation import generate_answer
from ragchunk.llm import build_gemini_generate_fn
from ragchunk.store.in_memory import InMemoryVectorStore
from ragchunk.store.indexer import Indexer

st.set_page_config(page_title="Adaptive RAG Demo", layout="wide")

if "indexer" not in st.session_state:
    st.session_state.indexer = None
if "embed_fn" not in st.session_state:
    st.session_state.embed_fn = None
if "generate_fn" not in st.session_state:
    st.session_state.generate_fn = None
if "indexed_log" not in st.session_state:
    st.session_state.indexed_log = []
if "indexed_filenames" not in st.session_state:
    st.session_state.indexed_filenames = set()
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = None

def get_providers():
    if st.session_state.embed_fn is not None and st.session_state.generate_fn is not None:
        return st.session_state.embed_fn, st.session_state.generate_fn, None

    embed_fn = build_gemini_embed_fn()
    generate_fn = build_gemini_generate_fn()

    if embed_fn is None or generate_fn is None:
        return None, None, (
            "Couldn't initialize the Gemini client. Check that GOOGLE_API_KEY is set "
            "(sidebar) and that `google-genai` is installed (`pip install google-genai`)."
        )

    st.session_state.embed_fn = embed_fn
    st.session_state.generate_fn = generate_fn
    return embed_fn, generate_fn, None

def reset_session():
    st.session_state.indexer = None
    st.session_state.embed_fn = None
    st.session_state.generate_fn = None
    st.session_state.indexed_log = None
    st.session_state.indexed_filenames = set()
    st.session_state.last_result = None
    st.session_state.last_retrieved = None

with st.sidebar:
    st.header("Configuration")

    existing_key = os.environ.get("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "GOOGLE_API_KEY",
        value=existing_key,
        type="password",
        help="From Google AI Studio (aistudio.google.com/apikey). Used for both "
        "embeddings and generation. Never stored beyond this session.",
    )
    if api_key_input and api_key_input != existing_key:
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.session_state.embed_fn = None
        st.session_state.generate_fn = None

    st.divider()
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=5)
    expand_to_parent = st.checkbox("Expand results to parent section", value=True)

    st.divider()
    if st.button("Reset session (clear index)", use_container_width=True):
            reset_session()
            st.rerun()

st.title("Adaptive RAG Pipeline Demo")
st.caption(
    "Classification-driven chunking -> pluggable embedding & vector storage ->" \
    "cited, grounding-verified generation."
)

st.subheader("1. Upload & index documents")
uploaded_files = st.file_uploader(
    "Upload one or more documents",
    type=["txt", "md", "pdf", "docx"],
    accept_multiple_files=True,
)

col_index, col_status = st.columns([1, 3])
with col_index:
    index_clicked = st.button("Index documents", type="primary", disabled=not uploaded_files)

if index_clicked and uploaded_files:
    embed_fn, generate_fn, error = get_providers()
    if error:
        st.error(error)
    else:
        if st.session_state.indexer is None:
            pipeline = AdaptiveChunkingPipeline()
            st.session_state.indexer = Indexer(
                pipeline, embed_fn=embed_fn,
                vector_store=InMemoryVectorStore(),
                build_parent_child=True,
            )

        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_filenames]
        if not new_files:
            st.info("All uploaded files are already indexed in this session.")
        else:
            progress = st.progress(0.0, text="Indexing...")
            with tempfile.TemporaryDirectory() as tmp_dir:
                for i, uploaded_file in enumerate(new_files):
                    tmp_path = Path(tmp_dir) / uploaded_file.name
                    tmp_path.write_bytes(uploaded_file.getvalue())

                    result = st.session_state.indexer.pipeline.process_file(str(tmp_path))
                    chunk_count = st.session_state.indexer.index_chunks(result.chunks)

                    st.session_state.indexed_log.append({
                        "File": uploaded_file.name,
                        "Detected type": result.doc_type.value,
                        "Method": result.classification.method,
                        "Confidence": f"{result.classification.confidence:.2f}",
                        "Chunks": chunk_count,
                    })
                    st.session_state.indexed_filenames.add(uploaded_file.name)
                    progress.progress((i + 1) / len(new_files), text=f"Indexed {uploaded_file.name}")
            progress.empty()
            st.success(f"Indexed {len(new_files)} new document(s).")

if st.session_state.indexed_log:
    st.table(st.session_state.indexed_log)
else:
    with col_status:
        st.info("No documents indexed yet in this session.")

st.divider()

st.subheader("2. Ask a question")

question = st.text_input("Question", placeholder="e.g. What law governs this agreement?")
ask_clicked = st.button(
    "Ask", 
    type="primary",
    disabled=(st.session_state.indexer is None or len(st.session_state.indexed_filenames) == 0),
)

if st.session_state.indexer is None or len(st.session_state.indexed_filenames) == 0:
    st.caption("Index at least one document above before asking a question.")

if ask_clicked and question:
    embed_fn, generate_fn, error = get_providers()
    if error:
        st.error(error)
    else:
        with st.spinner("Retrieving and generating..."):
            retrieved = st.session_state.indexer.query(question, top_k=top_k, expand_to_parents=expand_to_parent)
            result = generate_answer(question, retrieved, generate_fn)
        st.session_state.last_result = result
        st.session_state.last_retrieved = retrieved

result = st.session_state.last_result
if result: 
    st.markdown("Answer")
    if result.grounded:
        st.success(result.answer)
    else:
        st.warning(result.answer)
        st.caption(
            "No citation markers were found in this answer - treat it with extra caution, " \
            "it may not be grounded in the retrieved sources."
        )

    if result.citations:
        st.markdown("Cited Sources:")
        for c in result.citations:
            st.markdown(f"- `[{c.index}]` **{c.doc_id}** -- {c.section_path or 'n/a'} (retrieval score {c.score:.3f})")

    cited_ids = {c.chunk_id for c in result.citations}
    unused = [s for s in result.sources_provided if s.chunk_id not in cited_ids]
    if unused:
        with st.expander(f"{len(unused)} retrieved source(s) provided but not cited"):
            for s in unused:
                st.markdown(f"- **{s.doc_id}** -- {s.section_path or 'n/a'} (retrieval score {s.score:.3f})")

    if st.session_state.last_retrieved:
        with st.expander("Full retrieved chunk text (debug view)"):
            for r in st.session_state.last_retrieved:
                chunk = r["chunk"]
                st.markdown(f"**[{chunk.doc_type}] {chunk.section_path or 'n/a'}** (score {r['score']:.3f})")
                st.text(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))
                st.divider()