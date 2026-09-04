"""
CurriculumAI - Streamlit Frontend Interface
High-contrast, modern, multi-tab academic companion for PDF curricula.
"""

import os
import json
import hashlib
from datetime import datetime
import streamlit as st

from backend import (
    get_api_key,
    process_uploaded_file,
    query_curriculum_with_sources,
    generate_curriculum_overview,
    generate_study_quiz
)

# Page Configuration
st.set_page_config(
    page_title="CurriculumAI • Smart PDF Reader",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-Contrast, Clean Modern Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-main: #0B0F17;
    --bg-card: #131B2A;
    --bg-sidebar: #0F1624;
    --border: rgba(255, 255, 255, 0.12);
    --primary: #10B981;
    --primary-grad: linear-gradient(135deg, #10B981 0%, #059669 100%);
    --text-1: #F8FAFC;
    --text-2: #CBD5E1;
    --text-muted: #94A3B8;
}

html, body, .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-1) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3, h4, h5, h6 { color: var(--text-1) !important; font-weight: 700 !important; }
p, span, label { color: var(--text-2); }

.app-header {
    background: linear-gradient(180deg, rgba(19, 27, 42, 0.95) 0%, rgba(11, 15, 23, 0.98) 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
    border-top: 3px solid #10B981;
}

.brand-badge {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34D399 !important;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 9999px;
    text-transform: uppercase;
    display: inline-block;
    margin-bottom: 0.5rem;
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.stat-label { font-size: 0.75rem; color: var(--text-muted) !important; text-transform: uppercase; font-weight: 600; }
.stat-val { font-size: 1.35rem; font-weight: 800; color: #FFFFFF !important; }
.stat-sub { font-size: 0.75rem; color: #10B981 !important; }

.stTabs [data-baseweb="tab-list"] {
    background-color: #0E1522;
    padding: 5px;
    border-radius: 10px;
    border: 1px solid var(--border);
}

.stTabs [aria-selected="true"] {
    background-color: #1E293B !important;
    color: #34D399 !important;
    border-radius: 6px;
}

.stButton > button[kind="primary"] {
    background: var(--primary-grad) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
}

.stButton > button:not([kind="primary"]) {
    background: #192333 !important;
    color: #F1F5F9 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.stTextInput input, .stTextArea textarea, [data-baseweb="select"] {
    background-color: #131C2C !important;
    color: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

[data-testid="stChatMessage"] {
    background-color: #121A28 !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 0.8rem !important;
}

.source-box {
    background: #0D131F;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 0.7rem 0.9rem;
    margin-top: 0.5rem;
    font-size: 0.85rem;
}

.empty-hero {
    background: linear-gradient(145deg, #131C2D 0%, #0D1422 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 2.2rem 1.8rem;
    text-align: center;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


def init_state():
    """Initializes default session states."""
    defaults = {
        "api_key": get_api_key() or "",
        "vector_store": None,
        "documents": [],
        "chunks": [],
        "stats": None,
        "document_name": None,
        "document_hash": None,
        "overview": None,
        "quiz": None,
        "messages": [],
        "selected_model": "gemini-3.6-flash",
        "temperature": 0.2,
        "top_k": 4,
        "max_pages": 30
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def reset_state():
    """Resets document state and conversation."""
    for key in ["vector_store", "documents", "chunks", "stats", "document_name", "document_hash", "overview", "quiz"]:
        st.session_state[key] = None
    st.session_state.messages = []


def index_document(file_source, name: str):
    """Processes and indexes a PDF document with page limit safeguards."""
    key = get_api_key(st.session_state.api_key)
    if not key:
        st.error("❌ Google Gemini API Key required. Please enter it in the sidebar.")
        return

    with st.status(f"⚡ Indexing '{name}'...", expanded=True) as status:
        try:
            status.update(label="📄 Parsing PDF & generating embeddings...", state="running")
            result = process_uploaded_file(
                file_source,
                api_key=key,
                max_pages=st.session_state.max_pages
            )
            
            st.session_state.vector_store = result["vector_store"]
            st.session_state.documents = result["documents"]
            st.session_state.chunks = result["chunks"]
            st.session_state.stats = result["stats"]
            st.session_state.document_name = name
            st.session_state.overview = None
            st.session_state.quiz = None
            st.session_state.messages = []

            stats = result["stats"]
            if stats.get("is_truncated"):
                status.update(
                    label=f"⚠️ '{name}' indexed (Limited to first {stats['total_pages']} of {stats['original_total_pages']} pages)!",
                    state="complete"
                )
                st.toast(
                    f"Indexed first {stats['total_pages']} of {stats['original_total_pages']} pages ({stats['total_chunks']} chunks)!",
                    icon="⚠️"
                )
            else:
                status.update(label=f"✅ '{name}' indexed successfully!", state="complete")
                st.toast(f"Indexed {stats['total_pages']} pages ({stats['total_chunks']} chunks)!", icon="🎉")
        except Exception as e:
            status.update(label="❌ Indexing failed", state="error")
            st.error(f"Error: {str(e)}")


def ask_question(question: str):
    """Sends question to RAG pipeline and stores conversation."""
    if not question.strip() or not st.session_state.vector_store:
        return

    key = get_api_key(st.session_state.api_key)
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})

    with st.spinner("🔍 Retrieving curriculum context & synthesizing answer..."):
        try:
            res = query_curriculum_with_sources(
                vector_store=st.session_state.vector_store,
                question=question,
                model_name=st.session_state.selected_model,
                temperature=st.session_state.temperature,
                k=st.session_state.top_k,
                api_key=key
            )
            st.session_state.messages.append({"role": "assistant", "content": res["answer"], "sources": res["sources"]})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}", "sources": []})


# --- APP INITIALIZATION ---
init_state()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 1.8rem;">📚</span>
        <div style="font-weight: 800; font-size: 1.2rem; color: #FFF;">CurriculumAI</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # 1. API Key
    st.markdown("### 🔑 API Key")
    api_key_input = st.text_input("Gemini API Key", value=st.session_state.api_key, type="password", placeholder="AIzaSy...")
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input.strip()

    if st.session_state.api_key:
        st.markdown('<span style="color: #10B981; font-size: 0.8rem; font-weight: 600;">🟢 Key Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color: #EF4444; font-size: 0.8rem; font-weight: 600;">🔴 API Key Missing</span>', unsafe_allow_html=True)

    st.divider()

    # 2. Upload / Sample Document
    st.markdown("### 📂 Curriculum PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        h = hashlib.sha256(uploaded.getvalue()).hexdigest()
        if h != st.session_state.document_hash and st.button("⚡ Index Uploaded PDF", type="primary", use_container_width=True):
            st.session_state.document_hash = h
            index_document(uploaded, uploaded.name)
            st.rerun()

    demo_path = os.path.join(os.getcwd(), "unsupervised_learning.pdf")
    if os.path.exists(demo_path) and st.button("🚀 Load Demo PDF", use_container_width=True):
        with open(demo_path, "rb") as f:
            st.session_state.document_hash = hashlib.sha256(f.read()).hexdigest()
        index_document(demo_path, "unsupervised_learning.pdf")
        st.rerun()

    if st.session_state.document_name:
        st.caption(f"**Loaded:** `{st.session_state.document_name}`")
        if st.button("🗑️ Clear Document", use_container_width=True):
            reset_state()
            st.rerun()

    st.divider()

    # 3. Model Parameters & Processing Limits
    with st.expander("⚙️ RAG Settings"):
        st.session_state.selected_model = st.selectbox(
            "Model", ["gemini-3.6-flash", "gemini-3.6-pro", "gemini-3-flash", "gemini-3-pro"], index=0
        )
        st.session_state.max_pages = st.slider(
            "Max Pages to Index",
            min_value=5,
            max_value=100,
            value=st.session_state.max_pages,
            step=5,
            help="Limits the maximum number of PDF pages to embed. Prevents memory exhaustion, payload crashes, and API rate limits on large files."
        )
        st.session_state.top_k = st.slider("Top Chunks (k)", 1, 8, st.session_state.top_k)
        st.session_state.temperature = st.slider("Temperature", 0.0, 0.7, st.session_state.temperature, 0.05)


# --- MAIN HEADER ---
st.markdown("""
<div class="app-header">
    <div class="brand-badge">⚡ Academic AI Companion</div>
    <div style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; line-height: 1.1;">PDF Curriculum Reader & Study Assistant</div>
    <p style="color: #94A3B8; font-size: 1rem; margin-top: 0.3rem;">
        Transform textbooks, lecture notes & academic curricula into interactive, cited study sessions.
    </p>
</div>
""", unsafe_allow_html=True)

# --- STATS BAR ---
if st.session_state.stats and st.session_state.document_name:
    s = st.session_state.stats
    c1, c2, c3, c4 = st.columns(4)
    if s.get("is_truncated"):
        c1.markdown(f'<div class="stat-card"><span class="stat-label">Pages</span><span class="stat-val">{s["total_pages"]} <span style="font-size: 0.85rem; color: #F59E0B; font-weight: 600;">/ {s["original_total_pages"]}</span></span><span class="stat-sub" style="color: #F59E0B !important;">⚠️ Capped (Max {s["total_pages"]})</span></div>', unsafe_allow_html=True)
    else:
        c1.markdown(f'<div class="stat-card"><span class="stat-label">Pages</span><span class="stat-val">{s["total_pages"]}</span><span class="stat-sub">Parsed</span></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card"><span class="stat-label">Chunks</span><span class="stat-val">{s["total_chunks"]}</span><span class="stat-sub">FAISS Index</span></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-card"><span class="stat-label">Est. Reading</span><span class="stat-val">{s["est_reading_minutes"]} min</span><span class="stat-sub">~{s["total_words"]:,} words</span></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="stat-card"><span class="stat-label">Engine</span><span class="stat-val" style="font-size: 1.1rem; color: #38BDF8 !important;">{st.session_state.selected_model.replace("gemini-", "")}</span><span class="stat-sub">k={st.session_state.top_k} | T={st.session_state.temperature}</span></div>', unsafe_allow_html=True)
    st.write("")

    if s.get("is_truncated"):
        st.info(
            f"ℹ️ **Page Limit Active:** This PDF contains **{s['original_total_pages']} total pages**, but processing was capped at the first **{s['total_pages']} pages** to avoid API rate limits and keep embedding fast and manageable. You can adjust the **Max Pages to Index** slider in RAG Settings and re-index if needed.",
            icon="⚡"
        )


# --- MAIN CONTENT ---
if not st.session_state.document_name:
    st.markdown("""
    <div class="empty-hero">
        <div style="font-size: 2.8rem; margin-bottom: 0.5rem;">📖</div>
        <h3 style="color: #FFFFFF !important; margin-bottom: 0.5rem;">No Curriculum Loaded</h3>
        <p style="color: #94A3B8; max-width: 500px; margin: 0 auto 1.2rem auto;">
            Upload your course syllabus or PDF in the sidebar, or click below to launch the sample document.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if os.path.exists(demo_path) and st.button("🚀 Load Sample Curriculum (Unsupervised Learning PDF)", type="primary", use_container_width=True):
        with open(demo_path, "rb") as f:
            st.session_state.document_hash = hashlib.sha256(f.read()).hexdigest()
        index_document(demo_path, "unsupervised_learning.pdf")
        st.rerun()
else:
    tab_chat, tab_overview, tab_quiz, tab_explorer = st.tabs([
        "💬 Interactive Tutor", "📑 Syllabus & Overview", "🎯 Practice Quiz", "🔍 Chunk Inspector"
    ])

    # 1. Interactive Chat
    with tab_chat:
        st.markdown("#### 💡 Quick Prompts")
        qc1, qc2, qc3, qc4 = st.columns(4)
        action = None
        if qc1.button("📌 Core Topics", use_container_width=True):
            action = "Please provide a comprehensive summary of the main topics and key principles in this curriculum."
        if qc2.button("🔑 Key Formulas & Terms", use_container_width=True):
            action = "What are the most important definitions, formulas, and technical terminologies introduced in this document?"
        if qc3.button("⚖️ Compare Algorithms", use_container_width=True):
            action = "Compare the main algorithms or methods discussed in this curriculum, including their pros, cons, and use cases."
        if qc4.button("📝 Study Guide", use_container_width=True):
            action = "Create a high-yield exam preparation study guide with bullet points and common pitfalls to avoid."

        if action:
            ask_question(action)
            st.rerun()

        st.divider()

        if not st.session_state.messages:
            st.info("👋 Ask anything about your curriculum using the chat input below or click one of the quick prompts above!")
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🤖"):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        with st.expander(f"📍 Verified Sources ({len(msg['sources'])} page references)", expanded=False):
                            for idx, src in enumerate(msg["sources"], 1):
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong style="color: #38BDF8;">Reference #{idx} • Page {src['page']}</strong>
                                    <div style="color: #CBD5E1; font-style: italic; margin-top: 2px;">"{src['snippet']}"</div>
                                </div>
                                """, unsafe_allow_html=True)

            col_md, col_json, col_clear = st.columns([1, 1, 1])
            with col_md:
                st.download_button(
                    "📥 Export (.MD)",
                    data="\n\n".join([f"### {m['role'].upper()}:\n{m['content']}" for m in st.session_state.messages]),
                    file_name=f"{st.session_state.document_name}_notes.md",
                    use_container_width=True
                )
            with col_json:
                st.download_button(
                    "📥 Export (.JSON)",
                    data=json.dumps({"document": st.session_state.document_name, "messages": st.session_state.messages}, indent=2),
                    file_name=f"{st.session_state.document_name}_chat.json",
                    use_container_width=True
                )
            with col_clear:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

        query = st.chat_input("Ask a question about this curriculum...")
        if query:
            ask_question(query)
            st.rerun()

    # 2. Syllabus & Overview
    with tab_overview:
        st.markdown("### 📑 AI Curriculum Overview")
        if st.session_state.overview:
            st.markdown(st.session_state.overview)
            if st.button("🔄 Regenerate Overview"):
                st.session_state.overview = None
                st.rerun()
        else:
            if st.button("✨ Generate Curriculum Breakdown", type="primary"):
                with st.spinner("Analyzing curriculum structure..."):
                    try:
                        st.session_state.overview = generate_curriculum_overview(
                            st.session_state.documents,
                            st.session_state.selected_model,
                            get_api_key(st.session_state.api_key)
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {str(e)}")

        st.divider()
        st.markdown("#### 📖 Page Previews")
        if st.session_state.stats and st.session_state.stats.get("page_previews"):
            for p in st.session_state.stats["page_previews"]:
                with st.expander(f"Page {p['page_number']} Preview ({p['char_count']} chars)"):
                    st.text(p["preview"])

    # 3. Practice Quiz
    with tab_quiz:
        st.markdown("### 🎯 Knowledge Check & Practice Quiz")
        q_cols = st.columns([1, 2])
        num_q = q_cols[0].select_slider("Questions", [3, 5, 8, 10], 5)
        if q_cols[1].button("🎓 Generate Practice Quiz", type="primary", use_container_width=True):
            with st.spinner("Generating quiz questions..."):
                try:
                    st.session_state.quiz = generate_study_quiz(
                        st.session_state.documents,
                        st.session_state.selected_model,
                        num_q,
                        get_api_key(st.session_state.api_key)
                    )
                except Exception as e:
                    st.error(f"Failed: {str(e)}")

        if st.session_state.quiz:
            st.markdown("---")
            st.markdown(st.session_state.quiz)

    # 4. Vector Store Explorer
    with tab_explorer:
        st.markdown("### 🔍 Vector Store & Chunk Inspector")
        chunks = st.session_state.chunks
        st.caption(f"Total Indexed Chunks: `{len(chunks)}` (1000 char windows with 200 char overlap)")
        filter_kw = st.text_input("Filter Chunks by Keyword", placeholder="e.g. clustering, pca, loss...")
        matched = [(i, c) for i, c in enumerate(chunks) if not filter_kw or filter_kw.lower() in c.page_content.lower()]
        st.caption(f"Showing {min(len(matched), 20)} of {len(matched)} matching chunks:")
        for idx, chunk in matched[:20]:
            with st.expander(f"Chunk #{idx + 1} • Page {chunk.metadata.get('page', 0) + 1} ({len(chunk.page_content)} chars)"):
                st.code(chunk.page_content, language="text")
