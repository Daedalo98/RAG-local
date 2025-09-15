import os, re, json, shutil, subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests

from utils.parse_utils import ingest_folder, build_context_listing
from utils.report_utils import build_evidence_report
from utils.annotate_utils import annotate_sources_spans
from contextlib import nullcontext

# --------- CONFIG: "average" preset ----------
CFG = {
    # ...
    "ollama_model": "llama3:8b",
    "ollama_url": "http://localhost:11434",
    "chunk_tokens": 600,
    "overlap": 100,
    "final_ctx": 8,       # was 5
    "retriever_k": 12,    # a small recall bump
    "rerank_pool": 20,
    "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "rerank_model_name": "BAAI/bge-reranker-base",
    "max_ctx_chars_per_passage": 1200
}

INGEST_DIR = Path("ingest")
INDEX_DIR  = Path("index")
OUT_DIR    = Path("out")
OUT_ANS    = OUT_DIR / "answers"
OUT_ANN    = OUT_DIR / "annotated"
for p in (INDEX_DIR, OUT_ANS, OUT_ANN):
    p.mkdir(parents=True, exist_ok=True)

# --------- PROMPT ----------
ANSWER_PROMPT = """Write a clear, cohesive, and explanatory answer using ONLY the information in CONTEXT.
- Do NOT include citations, filenames, chunk ids, or any reference to sources in the text.
- Do NOT list bullets like “From the notes”. Compose a single narrative (or short sections if the question has multiple parts).
- If the question has multiple parts, answer each part in smooth prose within the same response.
- If at least one relevant statement exists in CONTEXT, DO NOT say “I don’t know…”. Answer using what’s there.
- Only if nothing relevant exists in CONTEXT for the entire question, say exactly: "I don’t know based on the provided documents."
- Keep it concise unless definitions or process steps are requested; 3–8 sentences are usually enough.
- Quote definitions or key terms exactly if present in the context.
- If the question has multiple parts, answer each in its own short section with a brief bold label (e.g., **What the notes say:** …, **Definition:** …), but do NOT include any citations, filenames, or chunk ids.
- Prefer direct, assertive phrasing when the CONTEXT supports the claim.
- If the CONTEXT has conflicting information, summarize the range of views in a balanced way.


QUESTION:
{question}

CONTEXT:
{context}
"""


# --- Multi-query generation via Ollama ---
def generate_subqueries(question: str, n: int = 5) -> list[str]:
    prompt = f"""Reformulate the user question into {n} diverse, specific search queries.
Return ONLY a JSON list of strings, no commentary.

Question: {question}"""
    raw = ollama_complete(prompt, model=CFG["ollama_model"], temperature=0.2)
    import json
    try:
        queries = json.loads(raw)
        if isinstance(queries, list):
            return [q for q in queries if isinstance(q, str) and q.strip()]
    except Exception:
        pass
    # fallback: simple perturbations
    base = question.strip()
    return [base, f'"{base}"', base + " details", base + " key facts", base + " summary"]

# --- Reciprocal Rank Fusion (RRF) ---
def rrf_fuse(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    ranked_lists: list of ranked candidate lists; each candidate has a stable 'id' key (we'll make one).
    Returns fused ranking (list of candidate dicts).
    """
    from collections import defaultdict
    scores = defaultdict(float)
    by_id = {}
    for rl in ranked_lists:
        for rank, item in enumerate(rl):
            cid = item.get("_cid")
            if not cid:
                cid = f"{item['source_path']}#{item['chunk_id']}"
                item["_cid"] = cid
            by_id[cid] = item
            scores[cid] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [by_id[cid] | {"rrf_score": s} for cid, s in fused]

# --- One retrieval for a single query (uses your hybrid + rerank) ---
def retrieve_single(query: str, index, embed_model, docs, bm25, top_pool=CFG["rerank_pool"], final_k=CFG["final_ctx"]):
    cands = hybrid_candidates(
        query, index, embed_model, docs, bm25,
        k_vec=CFG["retriever_k"], k_bm25=CFG["retriever_k"], rerank_pool=top_pool
    )
    rr_tok, rr_model = make_reranker()
    reranked = rerank(query, cands, rr_tok, rr_model, top_k=final_k)
    # stamp a stable candidate id for fusion
    for r in reranked:
        r["_cid"] = f"{r['source_path']}#{r['chunk_id']}"
    return reranked

# --- Fusion over multi-queries ---
def fusion_retrieve(question: str, index, embed_model, docs, bm25, n_subq: int = 5, final_k: int = None):
    subqs = generate_subqueries(question, n_subq)
    per_list = []
    for sq in subqs:
        per_list.append(retrieve_single(sq, index, embed_model, docs, bm25))
    fused = rrf_fuse(per_list, k=60)
    if final_k is None:
        final_k = CFG["final_ctx"]
    return fused[:final_k], subqs

# --- Decompose answer into atomic claims ---
def decompose_answer_to_claims(answer: str, max_claims: int = 6) -> list[str]:
    prompt = f"""Extract up to {max_claims} short, atomic claims from the ANSWER.
Return ONLY a JSON list of strings, each <= 25 words, no commentary.

ANSWER:
{answer}"""
    raw = ollama_complete(prompt, model=CFG["ollama_model"], temperature=0.0)
    import json
    try:
        claims = json.loads(raw)
        if isinstance(claims, list):
            return [c for c in claims if isinstance(c, str) and c.strip()]
    except Exception:
        pass
    # fallback: sentence split
    import re
    parts = re.split(r'(?<=[\.\!\?])\s+', answer)
    return [p.strip() for p in parts if p.strip()][:max_claims]

# --- For each claim, re-retrieve + pick exact span inside the best passage ---
def pick_best_span(passage_text: str, claim: str) -> str:
    # choose the sentence(s) in passage with max token overlap with the claim
    import re
    pt = passage_text
    sents = re.split(r'(?<=[\.\!\?])\s+', pt)
    def tokset(s): 
        return {w.lower() for w in re.findall(r"[a-zA-Z0-9]+", s)}
    cs = tokset(claim)
    best, best_score = "", -1.0
    for s in sents:
        ps = tokset(s)
        if not ps: 
            continue
        jacc = len(ps & cs) / max(1, len(ps | cs))
        if jacc > best_score:
            best, best_score = s, jacc
    # If nothing useful, fallback to first 160 chars
    return best.strip() if best else pt[:160].strip()

def attribute_answer(answer: str, index, embed_model, docs, bm25) -> dict:
    """
    Returns mapping:
      { source_path: { 'spans': [ {'text': span_text, 'chunk_id': int, 'page': int|None } ] } }
    """
    claims = decompose_answer_to_claims(answer)
    by_src = {}
    # reuse the standard reranker
    rr_tok, rr_model = make_reranker()

    # put reranker on GPU if available (match your rerank() behavior)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rr_model.to(device)
    if device.type == "cuda":
        rr_model.half()

    for claim in claims:
        # multi-query per claim, then fuse
        subqs = [claim, f'"{claim}"', f'{claim} key details']
        per_list = [retrieve_single(sq, index, embed_model, docs, bm25) for sq in subqs]
        fused = rrf_fuse(per_list, k=60)[:4]
        if not fused:
            continue

        # rerank the small fused set against the claim (GPU-aware)
        texts = [c["text"] for c in fused]
        batch = rr_tok([claim]*len(texts), texts, padding=True, truncation=True,
                       return_tensors="pt", max_length=512)
        batch = {k: v.to(device) for k, v in batch.items()}

        use_autocast = (device.type == "cuda" and rr_model.dtype == torch.float16)
        from contextlib import nullcontext
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_autocast else nullcontext()

        with torch.inference_mode(), amp_ctx:
            scores = rr_model(**batch).logits.squeeze(-1)

        best_idx = int(torch.argmax(scores).item())
        best_passage = fused[best_idx]

        span = pick_best_span(best_passage["text"], claim)
        src = best_passage["source_path"]
        entry = by_src.setdefault(src, {"spans": []})
        entry["spans"].append({
            "text": span,
            "chunk_id": best_passage["chunk_id"],
            "page": best_passage.get("page")
        })

    return by_src

# --------- CORE HELPERS ----------
def build_or_load_index(docs, embed_model):
    X_path = INDEX_DIR / "embeddings.npy"
    D_path = INDEX_DIR / "docs.jsonl"
    I_path = INDEX_DIR / "faiss.index"

    if X_path.exists() and D_path.exists() and I_path.exists():
        X = np.load(X_path)
        index = faiss.read_index(str(I_path))
        with open(D_path, "r", encoding="utf-8") as f:
            docs_loaded = [json.loads(line) for line in f]
        return index, X, docs_loaded

    texts = [d["text"] for d in docs]
    X = embed_model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=256,   # bigger batches for GPU
        normalize_embeddings=True,
        show_progress_bar=True
    )

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    np.save(X_path, X)
    with open(D_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    faiss.write_index(index, str(I_path))
    return index, X, docs

def build_bm25(docs):
    corpus = [d["text"].split() for d in docs]
    return BM25Okapi(corpus)

def hybrid_candidates(query, index, embed_model, docs, bm25, k_vec=12, k_bm25=12, rerank_pool=20):
    # dense
    qv = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(qv, k_vec)
    cand_dense = [(int(i), float(D[0][j])) for j, i in enumerate(I[0])]

    # lexical
    scores = bm25.get_scores(query.split())
    cand_lex = sorted([(i, float(scores[i])) for i in range(len(docs))],
                      key=lambda x: x[1], reverse=True)[:k_bm25]

    # fuse with simple min-max + sum
    def minmax_norm(pairs):
        if not pairs: return {}
        vs = np.array([v for _, v in pairs])
        lo, hi = vs.min(), vs.max()
        if hi - lo < 1e-9:
            return {i: 0.0 for i, _ in pairs}
        return {i: (v - lo) / (hi - lo) for i, v in pairs}

    nd = minmax_norm(cand_dense)
    nl = minmax_norm(cand_lex)
    fused = {}
    for i, v in nd.items(): fused[i] = fused.get(i, 0) + v
    for i, v in nl.items(): fused[i] = fused.get(i, 0) + v

    top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:rerank_pool]
    return [docs[i] | {"hybrid_score": float(s)} for i, s in top]

def make_reranker():
    tok = AutoTokenizer.from_pretrained(CFG["rerank_model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(CFG["rerank_model_name"])
    return tok, model

def rerank(query, candidates, rr_tok, rr_model, top_k=5):
    if not candidates:
        return []

    pairs_q = [query] * len(candidates)
    pairs_t = [c["text"] for c in candidates]

    batch = rr_tok(
        pairs_q, pairs_t,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    # Move tokenized tensors to the same device as the model
    device = next(rr_model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}

    # Use inference_mode (faster than no_grad), and autocast if on CUDA+FP16
    use_autocast = (device.type == "cuda" and rr_model.dtype == torch.float16)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_autocast else nullcontext()

    with torch.inference_mode(), amp_ctx:
        scores = rr_model(**batch).logits.squeeze(-1)

    # Bring to CPU plain python list
    scores = scores.float().cpu().tolist()

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
    out = []
    for c, s in ranked:
        c2 = dict(c)
        c2["rerank_score"] = float(s)
        out.append(c2)
    return out


def ollama_complete(prompt, model=None, temperature=0.2):
    import json, subprocess, shlex, requests
    base = CFG.get("ollama_url", "http://localhost:11434").rstrip("/")
    model = model or CFG["ollama_model"]

    # --- Try HTTP /api/generate ---
    try:
        r = requests.post(
            f"{base}/api/generate",
            json={"model": model, "prompt": prompt, "temperature": temperature, "stream": False},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        if r.status_code != 404:
            r.raise_for_status()
    except requests.RequestException:
        pass  # fall through

    # --- Fallback: HTTP /api/chat ---
    try:
        r2 = requests.post(
            f"{base}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": temperature},
                "stream": False,
            },
            timeout=120,
        )
        if r2.status_code == 200:
            data = r2.json()
            return (data.get("message", {}).get("content") or data.get("content") or data.get("response") or "").strip()
        # if not 200, fall through
    except requests.RequestException:
        pass

    # --- Final fallback: pure local CLI (no HTTP) ---
    try:
        # Note: `ollama run` prints only the response text.
        cmd = f'ollama run {shlex.quote(model)}'
        proc = subprocess.run(cmd, input=prompt.encode("utf-8"),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, check=True)
        return proc.stdout.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        raise RuntimeError(
            "All Ollama paths failed (HTTP /api/generate, /api/chat, and CLI). "
            f"Model: {model}. Error: {e}\n"
            "Quick checks:\n"
            "  1) curl http://localhost:11434/api/version\n"
            "  2) curl http://localhost:11434/api/tags\n"
            "  3) echo 'hi' | ollama run llama3:8b\n"
        )

def answer(question: str):
    # 1) ingest
    docs = ingest_folder(
        INGEST_DIR,
        chunk_tokens=CFG["chunk_tokens"],
        overlap=CFG["overlap"],
        max_ctx_chars=CFG["max_ctx_chars_per_passage"]
    )
    if not docs:
        raise RuntimeError("No documents found in ./ingest. Add files and retry.")

    # 2) build/load index
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(CFG["embed_model_name"], device=device)
    index, _, docs = build_or_load_index(docs, embed_model)

    # 3) BM25
    bm25 = build_bm25(docs)

    qnorm = question.strip()
    # If the query looks like two+ words, add a quoted phrase variant to help BM25
    q_for_bm25 = qnorm
    if " " in qnorm:
        q_for_bm25 = f'{qnorm} "{qnorm}"'

    # Pass q_for_bm25 into your BM25 scoring (simple tweak: split on spaces is fine)
    # If you use it inside build_bm25/hybrid, just replace the call site:
    # 4–5) Multi-query + Fusion retrieve -> final topk
    topk, used_subqueries = fusion_retrieve(question, index, embed_model, docs, bm25, n_subq=5, final_k=CFG["final_ctx"])

    for p in topk:
        txt = p["text"]
        m = re.split(r'(?<=[\.\!\?])\s+', txt)
        p["snippet"] = (m[0] if m and len(m[0]) >= 30 else txt[:200]).strip()

    # 6) LLM
    context_str = build_context_listing(topk)
    prompt = ANSWER_PROMPT.format(question=question, context=context_str)
    resp = ollama_complete(prompt)
    # Post-answer: attribute exact evidence spans per source
    attributed = attribute_answer(resp, index, embed_model, docs, bm25)


    # 7) annotate sources
    annotations = annotate_sources_spans(attributed, OUT_ANN)

    # 8) save bundle
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ans_path = OUT_ANS / f"answer-{stamp}.json"
    bundle_citations = []
    for p in topk:
        ann = annotations.get(p["source_path"], {})
        bundle_citations.append({
            "source_path": p["source_path"],
            "source_name": p["source_name"],
            "chunk_id": p["chunk_id"],
            "page": p.get("page"),
            "rerank_score": p.get("rerank_score"),
            "annotated_copy": ann.get("annotated_copy"),
            "spans": (ann.get("spans") or []),
        })

    bundle = {
    "question": question,
    "answer": resp,
    "citations": bundle_citations
    }
    ans_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build a click-to-open evidence report next to the JSON
    report_path = build_evidence_report(
        question, resp, bundle_citations, OUT_ANS, OUT_ANN
    )
    print(f"Evidence report: {report_path}")

    rp = str(report_path.resolve())
    try:
        # Prefer wslview if available
        if shutil.which("wslview"):
            subprocess.Popen(["wslview", rp])
        else:
            # Fallback to explorer.exe with Windows path
            winpath = subprocess.check_output(["wslpath", "-w", rp], text=True).strip()
            subprocess.Popen(["explorer.exe", winpath])
    except Exception as e:
        print(f"[WARN] Could not auto-open UI: {e}\nOpen manually: explorer.exe \"$(wslpath -w \"{rp}\")\"")


    ans_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAnswer saved: {ans_path}\n")
    print(resp)
    print("\nCitations:")
    for c in bundle["citations"]:
        print(f"- {c['source_name']}#{c['chunk_id']} -> annotated: {c['annotated_copy']}")
    return ans_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("question", type=str, help="Your question")
    args = ap.parse_args()
    answer(args.question)
