# RAG-Local (Ollama, GPU, Exact Evidence Highlights)

A local-first RAG pipeline that ingests your notes (Markdown, DOCX, PDF, HTML, TXT), retrieves with **multi-query + RRF fusion**, answers with a local **Ollama** model, and shows a browser **Evidence UI** with **exact highlighted spans** in the original documents.

## ✨ Features

* **Hybrid retrieval** (dense + BM25) with **multi-query reformulation** and **Reciprocal Rank Fusion** for robust recall.&#x20;
* **GPU acceleration** for **embeddings** (SentenceTransformers) and **reranking** (Transformers).&#x20;
* **Discursive answers** (no inline citations) while the UI presents sources with **precise highlights**:

  * PDF highlights via **PyMuPDF** (page-aware).&#x20;
  * DOCX highlights via **python-docx**.&#x20;
  * TXT/MD/HTML via sidecar evidence files with marked spans.&#x20;
* **Clean Evidence UI**: one HTML page listing sources; click to open a modal with spans + the highlighted copy.&#x20;
* **Obsidian-friendly**: just symlink or copy your vault into `ingest/`.&#x20;

---

## 🗂️ Repository layout

```
rag-local/
├─ ingest/                 # put your source documents here
├─ index/                  # cached embeddings & FAISS index (auto)
├─ out/
│  ├─ answers/             # JSON answers + evidence-report.html
│  └─ annotated/           # highlighted copies of sources
├─ utils/
│  ├─ parse_utils.py       # readers, cleaning, chunking, context render
│  ├─ annotate_utils.py    # exact-span highlighting (PDF/DOCX/TXT)
│  └─ report_utils.py      # Evidence UI (HTML)
└─ rag_local_annotated.py  # main entry point
```

* **`rag_local_annotated.py`**: config, multi-query & fusion retrieval, answer generation, post-answer claim attribution, UI build & auto-open.&#x20;
* **`utils/parse_utils.py`**: per-page PDF ingestion, chunking, filename injection for lexical recall, grouped context builder.&#x20;
* **`utils/annotate_utils.py`**: page-aware PDF highlights, DOCX highlights, whitespace-tolerant regex for text sources, span-driven annotator.&#x20;
* **`utils/report_utils.py`**: minimal JS/HTML modal viewer; lists spans above embedded highlighted copy.&#x20;

---

## 🖥️ Prerequisites

* **Python 3.10+** (WSL/Ubuntu recommended)
* **Ollama** installed locally (Windows or WSL). Make sure you’ve pulled a model, e.g.:

  ```bash
  ollama pull llama3:8b
  ```
* (Optional, recommended) **NVIDIA GPU** with drivers. Your Python stack already uses CUDA if `torch.cuda.is_available()` is `True`.&#x20;

> If Ollama runs on Windows (DirectML), GPU usage won’t show in `nvidia-smi`. That’s normal. If you want CUDA-visible usage, run Ollama inside WSL and stop the Windows service.

---

## 📦 Setup

```bash
# 1) Clone your repo
git clone git@github.com:<you>/rag-local.git
cd rag-local

# 2) Create venv & install
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## 📥 Add your documents

Put files under `ingest/`, e.g.:

```
ingest/
├─ Notes/
│  ├─ HTTP - HyperText Transfer Protocol.md
│  └─ API course 3 - how the web works.md
└─ PDFs/
   └─ Whitepaper.pdf
```

(You can symlink an Obsidian vault directory into `ingest/` if you prefer.)

---

## ⚙️ Config (defaults)

Key parameters live in `rag_local_annotated.py` → `CFG` (average profile). Defaults include:

* `chunk_tokens=600`, `overlap=100`
* `retriever_k=12`, `rerank_pool=20`, `final_ctx=8`
* `embed_model_name="sentence-transformers/all-MiniLM-L6-v2"`
* `rerank_model_name="BAAI/bge-reranker-base"`
* `ollama_model="llama3:8b"`, `ollama_url="http://localhost:11434"`&#x20;

You can bump **speed** or **depth** by adjusting these (see “Profiles” below).

---

## 🚀 Run

From the project root:

```bash
python rag_local_annotated.py "What do the notes say about HTTP? What is HTTP?"
```

What happens:

1. Ingest (PDF per-page), chunk & cache.&#x20;
2. Build or load FAISS + BM25.&#x20;
3. Multi-query generation → retrieve per sub-query → RRF fuse → rerank.&#x20;
4. Answer with Ollama (discursive; no inline citations).&#x20;
5. Post-answer: decompose into claims → re-retrieve → pick exact spans → annotate sources.&#x20;
6. Build **Evidence UI** and auto-open it (Windows or WSL).&#x20;

Outputs:

* `out/answers/answer-YYYYMMDD-HHMMSS.json` (answer + citations)
* `out/answers/evidence-report.html` (**open this**)
* `out/annotated/*` (highlighted copies)

If the browser didn’t open, use (WSL):

```bash
explorer.exe "$(wslpath -w "$(realpath out/answers/evidence-report.html)")"
```

---

## 🧠 Profiles: FAST vs DEEP

In `rag_local_annotated.py`, tune these for speed/accuracy trade-offs:

**FAST**

```python
chunk_tokens=900; overlap=40
retriever_k=10; rerank_pool=20; final_ctx=5
n_subq=3; max_claims=3
```

**DEEP**

```python
chunk_tokens=500; overlap=100
retriever_k=28; rerank_pool=80; final_ctx=10
n_subq=6; max_claims=6
```

(They map to the retrieval, fusion, and attribution logic in the main script.)&#x20;

---

## 🧩 Model choices (Ollama)

* **Balanced**: `mistral:7b-instruct`, `llama3:8b` (default)
* **Lighter/faster**: `phi3:mini`, `qwen2:7b-instruct`
* **Heavier**: `llama3.1:8b-instruct`, `mixtral:8x7b`

Change the model in `CFG["ollama_model"]`.&#x20;

---

## 🛠️ Troubleshooting

* **“No documents found in ./ingest”** → add files under `ingest/`.&#x20;
* **Ollama 404 on `/api/generate`** → code auto-falls back to `/api/chat`, then CLI. Ensure the model name exists (`ollama list`).&#x20;
* **GPU shows 0%**

  * If Ollama runs on **Windows**, it uses DirectML (won’t show in `nvidia-smi`). Use Task Manager’s GPU graphs or run Ollama inside WSL for CUDA visibility.
  * For Python, verify:

    ```python
    import torch; print(torch.cuda.is_available())
    ```

    The code already places embeddings and reranker on CUDA when available.&#x20;
* **UI opens “Documents” instead** → use absolute path conversion in WSL:

  ```bash
  explorer.exe "$(wslpath -w "$(realpath out/answers/evidence-report.html)")"
  ```
