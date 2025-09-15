# RAG Local (Ollama + Evidence Annotations)

Local Retrieval-Augmented Generation (RAG) that:
- Ingests PDFs, DOCX, TXT, MD, HTML
- Builds a FAISS index (dense) + BM25 (lexical) → **hybrid retrieve**
- Reranks with a cross-encoder (bge-reranker-base)
- Generates with **Ollama** (default: `llama3.1:8b`)
- **Highlights evidence inside original files** (PDF/DOCX) and sidecars for text formats

## Structure
