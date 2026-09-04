"""
CurriculumAI - Backend Service
Handles PDF parsing, chunking, FAISS vector indexing, and Gemini-powered RAG generation.
"""

import os
import tempfile
import warnings
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Suppress sunset warnings from langchain_community
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS

load_dotenv()

# Prompts
QA_PROMPT = PromptTemplate.from_template(
    """You are an expert academic tutor. Answer the question thoroughly based ONLY on the provided curriculum context.

Context:
{context}

Question:
{question}

Instructions:
1. Provide a well-structured answer using markdown, bullet points, and code/math blocks where helpful.
2. If the context does not contain enough information, state: "The uploaded curriculum does not contain enough information to answer this question."

Answer:"""
)

OVERVIEW_PROMPT = PromptTemplate.from_template(
    """You are an academic curriculum director. Provide a comprehensive, structured overview of this curriculum based on the excerpts.

Excerpts:
{context}

Include:
### 📌 Executive Summary
### 🎯 Key Learning Objectives & Themes
### 🗺️ Module / Topic Breakdown
### 💡 Prerequisites & Recommended Background

Overview:"""
)

QUIZ_PROMPT = PromptTemplate.from_template(
    """Generate {num_questions} high-yield conceptual practice questions with answers based on these curriculum excerpts.

Excerpts:
{context}

For each question:
1. Provide the Question with a concept tag (e.g. `[Concept: Topic]`).
2. Provide 4 options (A, B, C, D) or a conceptual challenge.
3. Provide the Correct Answer and a concise explanation.

Quiz:"""
)


def get_api_key(custom_key: Optional[str] = None) -> Optional[str]:
    """Resolves and synchronizes the active Gemini API key."""
    key = (custom_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        os.environ["GEMINI_API_KEY"] = key
        return key
    return None


def get_llm(model_name: str = "gemini-3.6-flash", temperature: float = 0.2, api_key: Optional[str] = None):
    """Factory for Gemini Chat LLM."""
    key = get_api_key(api_key)
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=key if key else None)


def get_embeddings(model_name: str = "gemini-embedding-2", api_key: Optional[str] = None):
    """Factory for Gemini Embeddings."""
    key = get_api_key(api_key)
    return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=key if key else None)


def format_docs(docs) -> str:
    """Formats retrieved document chunks with page references."""
    return "\n\n".join(f"--- Page {doc.metadata.get('page', 0) + 1} ---\n{doc.page_content}" for doc in docs)


DEFAULT_MAX_PAGES = 30


def extract_curriculum_stats(documents, chunks, original_total_pages: Optional[int] = None) -> Dict[str, Any]:
    """Calculates curriculum summary metrics and sample previews."""
    total_words = sum(len(doc.page_content.split()) for doc in documents)
    orig_pages = original_total_pages if original_total_pages is not None else len(documents)
    is_truncated = orig_pages > len(documents)
    return {
        "total_pages": len(documents),
        "original_total_pages": orig_pages,
        "is_truncated": is_truncated,
        "total_chunks": len(chunks),
        "total_words": total_words,
        "est_reading_minutes": max(1, round(total_words / 200)),
        "page_previews": [
            {
                "page_number": doc.metadata.get("page", idx) + 1,
                "char_count": len(doc.page_content),
                "preview": doc.page_content[:250].strip() + "..."
            }
            for idx, doc in enumerate(documents[:5])
        ]
    }


def process_uploaded_file(
    uploaded_file,
    api_key: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_pages: Optional[int] = DEFAULT_MAX_PAGES
) -> Dict[str, Any]:
    """
    Loads, chunks, and indexes a PDF into a FAISS vector store.
    Accepts a filepath string or a Streamlit UploadedFile.
    Limits parsed pages to max_pages to prevent memory overflow or embedding API limits on large PDFs.
    """
    is_path = isinstance(uploaded_file, str)
    temp_path = uploaded_file if is_path else None

    if not is_path:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

    try:
        # Load & split
        raw_docs = PyPDFLoader(temp_path).load()
        original_page_count = len(raw_docs)

        # Apply page cap to keep embedding manageable
        if max_pages and max_pages > 0 and original_page_count > max_pages:
            docs = raw_docs[:max_pages]
        else:
            docs = raw_docs

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # Embed & index
        vector_store = FAISS.from_documents(chunks, get_embeddings(api_key=api_key))
        stats = extract_curriculum_stats(docs, chunks, original_total_pages=original_page_count)

        return {
            "vector_store": vector_store,
            "documents": docs,
            "chunks": chunks,
            "stats": stats
        }
    finally:
        if not is_path and temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def query_curriculum_with_sources(
    vector_store,
    question: str,
    model_name: str = "gemini-3.6-flash",
    temperature: float = 0.2,
    k: int = 4,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Executes a RAG query and returns the answer with verified page citations."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    source_docs = retriever.invoke(question)

    llm = get_llm(model_name=model_name, temperature=temperature, api_key=api_key)
    prompt_str = QA_PROMPT.format(context=format_docs(source_docs), question=question)
    response = (llm | StrOutputParser()).invoke(prompt_str)

    sources = [
        {
            "page": doc.metadata.get("page", 0) + 1,
            "snippet": doc.page_content.strip()[:300] + ("..." if len(doc.page_content) > 300 else ""),
            "full_content": doc.page_content.strip()
        }
        for doc in source_docs
    ]

    return {"answer": response, "sources": sources}


def generate_curriculum_overview(documents, model_name: str = "gemini-3.6-flash", api_key: Optional[str] = None) -> str:
    """Generates a structured syllabus overview and learning objectives."""
    sample = "\n\n".join(
        f"--- Page {doc.metadata.get('page', 0) + 1} ---\n{doc.page_content[:1500]}"
        for doc in documents[::max(1, len(documents) // 6)][:8]
    )
    llm = get_llm(model_name=model_name, temperature=0.3, api_key=api_key)
    return (llm | StrOutputParser()).invoke(OVERVIEW_PROMPT.format(context=sample[:12000]))


def generate_study_quiz(documents, model_name: str = "gemini-3.6-flash", num_questions: int = 5, api_key: Optional[str] = None) -> str:
    """Generates high-yield practice quiz questions and answer keys."""
    sample = "\n\n".join(
        f"--- Page {doc.metadata.get('page', 0) + 1} ---\n{doc.page_content[:1200]}"
        for doc in documents[::max(1, len(documents) // 8)][:6]
    )
    llm = get_llm(model_name=model_name, temperature=0.3, api_key=api_key)
    return (llm | StrOutputParser()).invoke(QUIZ_PROMPT.format(context=sample[:10000], num_questions=num_questions))


# Legacy / standard aliases for backward compatibility
ensure_api_keys = get_api_key
LoadPDF = lambda path, max_pages=DEFAULT_MAX_PAGES: PyPDFLoader(path).load()[:max_pages] if max_pages else PyPDFLoader(path).load()
SplitPDF = lambda docs, chunk_size=1000, chunk_overlap=200: RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)
Create_Vector_Store = lambda chunks, api_key=None: FAISS.from_documents(chunks, get_embeddings(api_key=api_key))


def build_chain(vector_store, model_name: str = "gemini-3.6-flash", temperature: float = 0.2, k: int = 4, api_key: Optional[str] = None):
    """Builds a classic LCEL RAG chain."""
    llm = get_llm(model_name=model_name, temperature=temperature, api_key=api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )


if __name__ == "__main__":
    demo_pdf = "unsupervised_learning.pdf"
    if os.path.exists(demo_pdf):
        print("Indexing sample PDF to build chain...")
        result = process_uploaded_file(demo_pdf)
        chain = build_chain(result["vector_store"])
        print("\n=== LCEL Chain Graph (ASCII) ===")
        chain.get_graph().print_ascii()
        print("================================\n")