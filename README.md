<div align="center">

# 📚 CurriculumAI
**An AI-powered academic companion that turns textbooks, lecture notes, and syllabus PDFs into interactive, cited study sessions using Retrieval-Augmented Generation (RAG).**

<br>

[![Live Demo](https://img.shields.io/badge/LIVE_DEMO-STREAMLIT-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://m-bilal-habib-pdf-curriculum-reader-frontend-iswaya.streamlit.app/)
[![Python](https://img.shields.io/badge/PYTHON-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/MODEL-GEMINI-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![Framework](https://img.shields.io/badge/FRAMEWORK-LANGCHAIN-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com/)
[![Vector Store](https://img.shields.io/badge/VECTOR_STORE-FAISS-00A67E?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)

<br>

**Author:** [M Bilal Habib](https://github.com/m-bilal-habib)

</div>

---

## 🤖 Model Information

* **Model Used:** [`gemini-3.6-flash`](https://ai.google.dev/) (default) — also supports `gemini-3.6-pro`, `gemini-3-flash`, and `gemini-3-pro`.
* **Embeddings:** `gemini-embedding-2` / `models/embedding-001`, accessed via `langchain-google-genai`.
* **Model Type:** Google's instruction-tuned, long-context LLM family, accessed through the Gemini API.
* **Why this model?** Strong long-context comprehension and fast structured output generation — well suited for strictly grounded Q&A, syllabus synthesis, and quiz generation over dense academic PDFs.

---

## 🔑 Google Gemini API Key Setup

A Gemini API key is required to run the embedding and inference pipeline:

1. Create a free account at [Google AI Studio](https://aistudio.google.com/).
2. Go to **Get API Key** and generate a new key.
3. Add your key in **one of two ways**:
   * **Via `.env` file (Recommended):** Create a `.env` file in the root directory:
     ```env
     GOOGLE_API_KEY=your_gemini_api_key_here
     ```
   * **Via UI:** Paste your key directly into the sidebar text field inside the running Streamlit app.

---

## ✨ Main Features

* **💬 Interactive Academic Tutor (RAG Q&A):** Ask context-grounded questions about your curriculum, with strict grounding to uploaded content to minimize hallucinations and expandable source citations showing exact page numbers and snippets.
* **⚡ One-Click Quick Prompts:** Instant prompts for Core Topics, Key Formulas & Terms, Algorithm Comparison, and Exam Study Guide.
* **📑 Automated Syllabus & Curriculum Breakdown:** Synthesizes curriculum excerpts into structured executive summaries, learning objectives, module breakdowns, and prerequisite checklists — with live page preview snippets and document metrics.
* **🎯 AI Practice Quiz Generator:** Generates 3–10 configurable, high-yield conceptual questions with answer keys and detailed explanations.
* **🔍 Vector Store & Chunk Inspector:** Real-time search and inspection of indexed text chunks (1,000-char windows, 200-char overlap), with keyword filtering and page metadata verification.
* **📊 Document Analytics & Safeguards:** Live metrics for total pages, indexed chunks, estimated reading time, and word count, plus a configurable **Max Pages to Index** safeguard to prevent payload overload and rate-limit spikes.
* **📤 One-Click Export:** Export chat history and notes to Markdown (`.md`) or JSON.
* **🎨 Modern, High-Contrast UI:** Dark-mode interface styled with `Plus Jakarta Sans` and `JetBrains Mono`.

---

## 🚀 Quickstart

### 1. Clone & Set Up a Virtual Environment
```bash
cd "PDF Curriculum Reader"
python -m venv myenv
source myenv/bin/activate   # Windows: .\myenv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirement.txt
```

### 3. Configure Your API Key
Create a `.env` file in the root directory (see [API Key Setup](#-google-gemini-api-key-setup) above):
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Run the App
```bash
streamlit run frontend.py
```
Then open `http://localhost:8501` in your browser.

---

## 🗺️ Quick Walkthrough

1. **Load a Document** — drag and drop any academic PDF into the sidebar, or click **🚀 Load Demo PDF** to explore the bundled `unsupervised_learning.pdf`.
2. **Ask Questions** — use the **💬 Interactive Tutor** tab, or click any quick-prompt button.
3. **Inspect Citations** — click **📍 Verified Sources** beneath any answer to view exact pages and source snippets.
4. **Generate Study Materials** — visit **📑 Syllabus & Overview** for a structural outline, or **🎯 Practice Quiz** to test comprehension.
5. **Inspect Vectors** — visit **🔍 Chunk Inspector** to search and debug chunked representations.

---

## ⚙️ Configuration & RAG Settings

Customizable from the Streamlit sidebar under **⚙️ RAG Settings**:

| Setting | Default | Description |
| :--- | :--- | :--- |
| **Model** | `gemini-3.6-flash` | Active Gemini model (`gemini-3.6-flash`, `gemini-3.6-pro`, `gemini-3-flash`, `gemini-3-pro`). |
| **Max Pages to Index** | `30` | Safeguard cap for embedding large textbooks without hitting API rate limits or memory bottlenecks. |
| **Top Chunks (k)** | `4` | Number of most relevant document chunks retrieved per query. |
| **Temperature** | `0.20` | Controls model creativity (`0.0` for factual/strict, higher for creative responses). |

---

## 📂 Project Structure

```plaintext
PDF Curriculum Reader/
│
├── backend.py                 # Core RAG pipeline, indexing, prompt templates, & LLM services
├── frontend.py                # Streamlit UI with multi-tab interface and styling
├── requirement.txt            # Python dependencies
├── unsupervised_learning.pdf  # Sample demo PDF for instant testing
├── .env                       # API keys and environment variables (ignored in git)
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```

---

## 🛡️ License

This project is open source and available under the [MIT License](LICENSE).