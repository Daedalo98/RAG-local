import re, uuid, glob
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
from markdownify import markdownify as md2txt

# --------- READING ---------
def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Return list of (page_index, text) for a PDF."""
    reader = PdfReader(str(path))
    pages = []
    for i, p in enumerate(reader.pages):
        pages.append((i, p.extract_text() or ""))
    return pages

def _read_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    paras = [p.text for p in doc.paragraphs]
    return "\n".join(paras)

def _read_html(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")

def _read_md(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return md2txt(raw)

def load_any(path: Path) -> str:
    p = str(path).lower()
    if p.endswith(".pdf"):  return _read_pdf(path)
    if p.endswith(".docx"): return _read_docx(path)
    if p.endswith(".html") or p.endswith(".htm"): return _read_html(path)
    if p.endswith(".md"):   return _read_md(path)
    return _read_txt(path)

# --------- CLEAN / CHUNK ---------
def clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def chunk_text(text: str, chunk_tokens=600, overlap=80) -> List[str]:
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_tokens]
        if not chunk: break
        out.append(" ".join(chunk))
        i += max(1, chunk_tokens - overlap)
    return out

# --------- CONTEXT RENDER ---------
def build_context_listing(passages: List[Dict]) -> str:
    # Group by source for readability
    from collections import defaultdict
    g = defaultdict(list)
    for p in passages:
        g[(p['source_name'])].append(p)
    parts = []
    for src, items in g.items():
        items = sorted(items, key=lambda x: x['chunk_id'])
        for p in items:
            # keep chunks tidy
            txt = p['text'].strip()
            if len(txt) > 800:
                txt = txt[:800] + " ..."
            parts.append(f"[{p['source_name']}#{p['chunk_id']}] {txt}")
    return "\n\n".join(parts)

# --------- INGEST ---------
def ingest_folder(folder: Path, *, chunk_tokens=600, overlap=80, max_ctx_chars=1200) -> List[Dict]:
    docs: List[Dict] = []
    for path in folder.rglob("*"):
        if path.is_dir():
            continue
        lower = str(path).lower()
        if not any(lower.endswith(ext) for ext in (".pdf",".docx",".txt",".md",".html",".htm")):
            continue
        try:
            if lower.endswith(".pdf"):
                # --- PDF: chunk per-page and keep page number ---
                for page_idx, raw_text in _read_pdf_pages(path):
                    txt = clean_text(raw_text)
                    chunks = chunk_text(txt, chunk_tokens, overlap)
                    for j, ch in enumerate(chunks):
                        docs.append({
                            "id": f"{uuid.uuid4()}",
                            "text": ch[:max_ctx_chars],
                            "source_path": str(path),
                            "source_name": path.name,
                            "chunk_id": j,
                            "page": page_idx,          # <— keep page
                            "is_pdf": True
                        })
            else:
                # Non-PDF unchanged, but we inject filename to help retrieval
                raw = load_any(path)
                txt = clean_text(raw)
                chunks = chunk_text(txt, chunk_tokens, overlap)
                for j, ch in enumerate(chunks):
                    docs.append({
                        "id": f"{uuid.uuid4()}",
                        "text": f"[FILENAME {path.name}] " + ch[:max_ctx_chars],
                        "source_path": str(path),
                        "source_name": path.name,
                        "chunk_id": j,
                        "is_pdf": False
                    })
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
    return docs

