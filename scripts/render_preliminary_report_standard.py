"""Use the packaged rasterizer with an already exported Word PDF on Windows.

The packaged LibreOffice executable is unavailable in this runtime. The PDF is
exported by export_report_word_pdf.ps1 with Word hidden. No managed files change.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path('C:/Users/15037/.cache/codex-runtimes/codex-primary-runtime/dependencies')
SKILL = Path('C:/Users/15037/.codex/plugins/cache/openai-primary-runtime/documents/26.1025.11240/skills/documents')
DOCX = ROOT / 'docs/初赛单卡算子开发报告_标准重点版_20261028.docx'
OUT = ROOT / 'logs/report-standard-qa'
PDF = OUT / 'report-standard.pdf'
os.environ['PATH'] = str(RUNTIME / 'native/poppler/Library/bin') + os.pathsep + os.environ['PATH']
spec = importlib.util.spec_from_file_location('packaged_render_docx', SKILL / 'render_docx.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def use_exported_pdf(doc_path, user_profile, convert_tmp_dir, stem, verbose):
    assert Path(doc_path).resolve() == DOCX.resolve()
    assert PDF.stat().st_mtime >= DOCX.stat().st_mtime, 'Word PDF is stale'
    destination = Path(convert_tmp_dir) / (stem + '.pdf')
    shutil.copyfile(PDF, destination)
    return str(destination), 'Microsoft Word hidden PDF export; packaged Poppler rasterization'


renderer.convert_to_pdf = use_exported_pdf
pages = renderer.rasterize(str(DOCX), str(OUT / 'pages'), 160, verbose=False, emit_pdf=False)
from pypdf import PdfReader
pdf_pages = PdfReader(PDF).pages
summary = {
    'renderer': 'Word hidden PDF export plus canonical render_docx.py rasterize using bundled Poppler',
    'page_count': len(pages),
    'pages': [{'page': i+1, 'path': path, 'text_length': len(pdf_pages[i].extract_text())}
              for i, path in enumerate(pages)],
    'visual_review': 'pending',
}
(OUT / 'render.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=True))
