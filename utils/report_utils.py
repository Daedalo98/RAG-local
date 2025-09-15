# utils/report_utils.py
from pathlib import Path
import re, html
from typing import List, Dict, Optional
from docx import Document as DocxDocument

def _write_html(path: Path, html_text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")

def _base_page(title: str, body_html: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
  h1 {{ font-size: 20px; margin-bottom: 12px; }}
  .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
  .btn {{ display:inline-block; padding:8px 12px; border-radius:8px; border:1px solid #ccc; cursor:pointer; background:#f7f7f7; }}
  .meta {{ color:#555; font-size: 12px; }}
  mark {{ background: #fff3a3; }}
  /* Modal */
  .modal {{ display:none; position:fixed; z-index:9999; left:0; top:0; width:100%; height:100%; background:rgba(0,0,0,.5); }}
  .modal-content {{ position:relative; background:#fff; margin:3% auto; padding:0; border-radius:12px; width:92%; height:90%; overflow:hidden; display:flex; flex-direction:column; }}
  .modal-header {{ padding:10px 14px; border-bottom:1px solid #eee; display:flex; justify-content:space-between; align-items:center; }}
  .modal-body {{ flex:1; display:flex; flex-direction:column; overflow:hidden; }}
  .evidence-spans {{ padding:12px; border-bottom:1px solid #eee; max-height:120px; overflow:auto; }}
  .evidence-spans p {{ margin:4px 0; font-size:14px; }}
  .close {{ cursor:pointer; font-size:20px; }}
  iframe {{ border:0; width:100%; flex:1; }}
  .preview-html {{ padding:16px; overflow:auto; height:100%; }}
</style>
</head>
<body>
{body_html}
<script>
function openModal(id) {{
  document.getElementById(id).style.display = 'block';
}}
function closeModal(id) {{
  document.getElementById(id).style.display = 'none';
}}
</script>
</body>
</html>"""

def build_evidence_report(question: str, answer_text: str, citations: List[Dict], out_answers_dir: Path, out_annotated_dir: Path) -> Path:
    """
    citations: List of dicts with keys:
      - source_path, source_name, chunk_id, annotated_copy, spans (list of {text,page,chunk_id})
    """
    tiles = []
    modals = []
    for i, c in enumerate(citations, 1):
        src = c["source_path"]
        spans = c.get("spans") or []
        preview_rel = None
        if c.get("annotated_copy"):
            # we already built annotated copies in out/annotated/
            relpath = Path(c["annotated_copy"]).relative_to(out_annotated_dir)
            preview_rel = str(Path("../annotated") / relpath)

        modal_id = f"modal_{i}"
        # spans list HTML
        span_html = ""
        if spans:
            items = "".join(f"<p><mark>{html.escape(sp['text'])}</mark></p>" for sp in spans if sp.get("text"))
            span_html = f"<div class='evidence-spans'><strong>Evidence spans:</strong>{items}</div>"

        tile = f"""
        <div class="card">
          <div><strong>{html.escape(c['source_name'])}</strong> <span class="meta">#{c['chunk_id']}</span></div>
          <div class="meta">{html.escape(src)}</div>
          <div style="margin-top:8px;">
            <button class="btn" onclick="openModal('{modal_id}')">Open highlighted source</button>
          </div>
        </div>"""

        if preview_rel:
            iframe_html = f"<iframe src='{html.escape(preview_rel)}'></iframe>"
        else:
            iframe_html = "<div class='preview-html'><p class='meta'>No annotated copy available.</p></div>"

        modal = f"""
        <div id="{modal_id}" class="modal" onclick="if(event.target===this)closeModal('{modal_id}')">
          <div class="modal-content">
            <div class="modal-header">
              <div><strong>{html.escape(c['source_name'])}</strong> <span class="meta">#{c['chunk_id']}</span></div>
              <div class="close" onclick="closeModal('{modal_id}')">&times;</div>
            </div>
            <div class="modal-body">
              {span_html}
              {iframe_html}
            </div>
          </div>
        </div>"""

        tiles.append(tile)
        modals.append(modal)

    body = f"""
    <h1>Evidence Report</h1>
    <div class="card"><div><strong>Question:</strong> {html.escape(question)}</div>
    <div style="margin-top:6px;"><strong>Answer:</strong> {html.escape(answer_text)}</div></div>
    {''.join(tiles)}
    {''.join(modals)}
    """

    report_path = out_answers_dir / "evidence-report.html"
    _write_html(report_path, _base_page("Evidence Report", body))
    return report_path
