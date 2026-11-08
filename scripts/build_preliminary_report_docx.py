"""Render the concise report source into an editable Word document."""
from pathlib import Path
import re
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'docs/preliminary-report-submission.md'
TARGET = ROOT/'docs/初赛单卡算子技术报告_20261028实测版.docx'


def font(style, size, east='宋体', latin='Calibri'):
    style.font.name=latin;style.font.size=Pt(size);style.font.color.rgb=RGBColor(0,0,0)
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),east)


def inline(paragraph, text):
    # This report has simple inline code and bold, with no embedded equations.
    for piece in re.split(r'(`[^`]+`|\*\*[^*]+\*\*)',text):
        if piece.startswith('`') and piece.endswith('`'):
            run=paragraph.add_run(piece[1:-1]);run.font.name='Consolas';run.font.size=Pt(10)
        elif piece.startswith('**') and piece.endswith('**'):
            paragraph.add_run(piece[2:-2]).bold=True
        else:paragraph.add_run(piece)


def table(doc, rows):
    t=doc.add_table(rows=1, cols=len(rows[0]));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    widths=([1.25,5.75] if len(rows[0])==2 else [.6,1.75,1.65,1.65,1.35])
    for col,width in zip(t.columns,widths):col.width=Inches(width)
    props=t._tbl.tblPr
    borders=OxmlElement('w:tblBorders')
    for side in ('top','left','bottom','right','insideH','insideV'):
        el=OxmlElement('w:'+side);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
    props.append(borders)
    for i,values in enumerate(rows):
        row=t.rows[0] if i==0 else t.add_row()
        trPr=row._tr.get_or_add_trPr();trPr.append(OxmlElement('w:cantSplit'))
        if i==0:trPr.append(OxmlElement('w:tblHeader'))
        for j,(cell,value,width) in enumerate(zip(row.cells,values,widths)):
            cell.width=Inches(width);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcPr=cell._tc.get_or_add_tcPr()
            margins=OxmlElement('w:tcMar')
            for side in ('top','left','bottom','right'):
                el=OxmlElement('w:'+side);el.set(qn('w:w'),'95');el.set(qn('w:type'),'dxa');margins.append(el)
            tcPr.append(margins)
            shading=OxmlElement('w:shd');shading.set(qn('w:fill'),'DFEAF2' if i==0 else 'FFFFFF');tcPr.append(shading)
            p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2)
            p.paragraph_format.line_spacing=1.08
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER if len(values)>2 or j==0 else WD_ALIGN_PARAGRAPH.LEFT
            inline(p,value)
            for run in p.runs:run.font.size=Pt(10.5);run.bold=i==0
    doc.add_paragraph().paragraph_format.space_after=Pt(1)


def main():
    doc=Document();sec=doc.sections[0]
    # Some runtime templates give Title an inherited decorative bottom border.
    for style in doc.styles:
        for border in list(style.element.iter(qn('w:pBdr'))):
            border.getparent().remove(border)
    sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.left_margin=sec.right_margin=Inches(.75)
    sec.top_margin=sec.bottom_margin=Inches(.65)
    sec.header_distance=sec.footer_distance=Inches(.3)
    font(doc.styles['Normal'],11)
    doc.styles['Normal'].paragraph_format.line_spacing=1.15
    doc.styles['Normal'].paragraph_format.space_after=Pt(6)
    font(doc.styles['Title'],18,'微软雅黑');doc.styles['Title'].paragraph_format.space_after=Pt(10)
    font(doc.styles['Heading 1'],13,'微软雅黑');doc.styles['Heading 1'].font.bold=True
    doc.styles['Heading 1'].paragraph_format.space_before=Pt(12)
    doc.styles['Heading 1'].paragraph_format.space_after=Pt(6)
    font(doc.styles['List Bullet'],11)
    doc.styles['List Bullet'].paragraph_format.space_after=Pt(4)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    r=footer.add_run('2026-10-28  单卡首轮实测稿  |  ');r.font.size=Pt(9)
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    lines=SOURCE.read_text(encoding='utf-8').splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',c) for c in row):rows.append(row)
                i+=1
            table(doc,rows);continue
        if line.startswith('```'):
            i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                p=doc.add_paragraph();p.paragraph_format.space_after=Pt(2)
                p.paragraph_format.line_spacing=1.05;p.paragraph_format.keep_with_next=True
                r=p.add_run(lines[i]);r.font.name='Consolas';r.font.size=Pt(10);i+=1
            p.paragraph_format.keep_with_next=False;i+=1;continue
        if line.startswith('# '):inline(doc.add_paragraph(style='Title'),line[2:])
        elif line.startswith('## '):inline(doc.add_paragraph(style='Heading 1'),line[3:])
        elif line.startswith('- '):inline(doc.add_paragraph(style='List Bullet'),line[2:])
        else:inline(doc.add_paragraph(),line)
        i+=1
    doc.core_properties.title='MiniMax H3 单卡算子优化阶段技术报告'
    doc.core_properties.subject='初赛单卡正确性与性能实测 首轮性能退化 队伍信息待补'
    doc.core_properties.author='';doc.core_properties.keywords='MiniMax H3, InfiniCore, 九齿, 单卡'
    doc.save(TARGET);print(TARGET)


if __name__=='__main__':main()
