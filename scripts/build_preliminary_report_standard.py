"""Build the focused report, embedding original screenshots without rewriting pixels."""
from pathlib import Path
import json
import re
from PIL import Image
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/preliminary-report-standard.md'
TARGET = ROOT / 'docs/初赛单卡算子开发报告_标准重点版_20261028.docx'


def set_font(style, size, east='宋体', latin='Calibri'):
    style.font.name = latin
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0, 0, 0)
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    fonts.set(qn('w:eastAsia'), east)


def inline(p, value, size=None):
    # Keep reference labels and relative paths in the portable Word report.
    value = re.sub(r'\[([^]]+)\]\(([^)]+)\)', r'\1（\2）', value)
    for piece in re.split(r'(`[^`]+`|\*\*[^*]+\*\*)', value):
        if piece.startswith('`') and piece.endswith('`'):
            run = p.add_run(piece[1:-1])
            run.font.name = 'Consolas'
            run.font.size = Pt(size or 9.5)
        elif piece.startswith('**') and piece.endswith('**'):
            run = p.add_run(piece[2:-2])
            run.bold = True
        else:
            run = p.add_run(piece)
        if size:
            run.font.size = Pt(size)


def add_table(doc, rows):
    width_map = {2: [1.25, 5.60], 3: [1.35, 3.55, 1.95],
                 4: [0.65, 2.15, 2.15, 1.90]}
    widths = width_map[len(rows[0])]
    table = doc.add_table(rows=1, cols=len(widths))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for col, width in zip(table.columns, widths):
        col.width = Inches(width)
    borders = OxmlElement('w:tblBorders')
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement('w:' + side)
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), '4')
        el.set(qn('w:color'), 'D9D9D9')
        borders.append(el)
    table._tbl.tblPr.append(borders)
    for i, values in enumerate(rows):
        row = table.rows[0] if i == 0 else table.add_row()
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement('w:cantSplit'))
        if i == 0:
            trpr.append(OxmlElement('w:tblHeader'))
        for j, (cell, value, width) in enumerate(zip(row.cells, values, widths)):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr = cell._tc.get_or_add_tcPr()
            margins = OxmlElement('w:tcMar')
            for side in ('top', 'left', 'bottom', 'right'):
                el = OxmlElement('w:' + side)
                el.set(qn('w:w'), '70')
                el.set(qn('w:type'), 'dxa')
                margins.append(el)
            tcpr.append(margins)
            shade = OxmlElement('w:shd')
            shade.set(qn('w:fill'), 'DBE8EF' if i == 0 else 'FFFFFF')
            tcpr.append(shade)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.line_spacing = 1.06
            inline(p, value, 10)
            if len(widths) == 4 or j == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                if i == 0:
                    run.bold = True
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = Pt(3)


def add_image(doc, path, alt, crop_top):
    # DrawingML crop only: the embedded image remains the exact original file.
    with Image.open(path) as src:
        width, height = src.size
    visible_height = height * (1 - crop_top / 100)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.line_spacing = 1
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    display_width = 6.2 if path.name.startswith('00_') else 6.85
    shape = p.add_run().add_picture(str(path), width=Inches(display_width),
                                   height=Inches(display_width * visible_height / width))
    shape._inline.docPr.set('descr', alt)
    if crop_top:
        fill = shape._inline.graphic.graphicData.pic.blipFill
        rect = OxmlElement('a:srcRect')
        rect.set('t', str(round(crop_top * 1000)))
        fill.insert(1, rect)


def main():
    doc = Document()
    for style in doc.styles:
        for border in list(style.element.iter(qn('w:pBdr'))):
            border.getparent().remove(border)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.left_margin = sec.right_margin = Mm(18)
    sec.top_margin = sec.bottom_margin = Mm(16)
    sec.header_distance = sec.footer_distance = Mm(7)
    for name in ('Normal', 'List Bullet', 'Caption'):
        set_font(doc.styles[name], 11 if name != 'Caption' else 10)
        pf = doc.styles[name].paragraph_format
        pf.line_spacing = 1.12
        pf.space_after = Pt(5)
    doc.styles['Caption'].font.italic = False
    doc.styles['Caption'].font.bold = False
    doc.styles['Caption'].paragraph_format.keep_with_next = False
    doc.styles['Caption'].paragraph_format.space_after = Pt(3)
    for prop in list(doc.styles['Caption'].element.iter(qn('w:bCs'))):
        prop.set(qn('w:val'), '0')
    set_font(doc.styles['Title'], 18, '微软雅黑')
    doc.styles['Title'].paragraph_format.space_after = Pt(9)
    for name, size in [('Heading 1', 13), ('Heading 2', 11.5)]:
        set_font(doc.styles[name], size, '微软雅黑')
        doc.styles[name].font.bold = True
        pf = doc.styles[name].paragraph_format
        pf.space_before = Pt(8 if name == 'Heading 1' else 5)
        pf.space_after = Pt(5)
        pf.keep_with_next = True
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    lines = SOURCE.read_text(encoding='utf-8').splitlines()
    i, crop_top, after_image, page_start = 0, 0, False, False
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line == '<!-- pagebreak -->':
            page_start = True
        elif line.startswith('<!-- docx-crop-top:'):
            crop_top = float(re.search(r': ([\d.]+)', line)[1])
        elif line.startswith('!['):
            m = re.fullmatch(r'!\[([^]]+)\]\((.+)\)', line)
            add_image(doc, SOURCE.parent / m[2], m[1], crop_top)
            crop_top, after_image = 0, True
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                row = [s.strip() for s in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+', s) for s in row):
                    rows.append(row)
                i += 1
            add_table(doc, rows)
            continue
        elif line.startswith('```'):
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            for n, value in enumerate(code):
                p = doc.add_paragraph()
                pf = p.paragraph_format
                pf.space_after = Pt(1)
                pf.line_spacing = 1.05
                pf.keep_with_next = n < len(code) - 1
                r = p.add_run(value)
                r.font.name = 'Consolas'
                r.font.size = Pt(9)
        elif line.startswith('# '):
            inline(doc.add_paragraph(style='Title'), line[2:])
        elif line.startswith('## '):
            p = doc.add_paragraph(style='Heading 1')
            if page_start:
                p.paragraph_format.page_break_before = True
                page_start = False
            inline(p, line[3:])
        elif line.startswith('### '):
            inline(doc.add_paragraph(style='Heading 2'), line[4:])
        elif line.startswith('- '):
            inline(doc.add_paragraph(style='List Bullet'), line[2:])
        else:
            p = doc.add_paragraph(style='Caption' if after_image else 'Normal')
            inline(p, line)
            after_image = False
        i += 1
    doc.core_properties.title = 'MiniMax H3 初赛单卡算子开发报告'
    doc.core_properties.subject = 'InfiniCore 算子开发与大赛服务器测试结果'
    doc.core_properties.author = ''
    doc.core_properties.keywords = 'InfiniCore, InfiniLM, MHA, 九齿, 门控残差'
    doc.save(TARGET)
    print(json.dumps({'output': str(TARGET), 'figures': len(doc.inline_shapes)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
