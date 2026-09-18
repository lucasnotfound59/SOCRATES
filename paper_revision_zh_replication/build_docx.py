"""Build the APA-formatted Word manuscript from the Markdown paper.

Pandoc converts the Markdown structures (tables, emphasis, links); this script
then imposes the formatting the submission requires: Times New Roman 12 pt,
1.5 line spacing, 1-inch margins, a page number in the header, APA heading
levels, a hanging-indent reference list, and tables and figures moved behind
the reference list as APA 7 prefers.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import docx
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Final_Paper_EN_Chinese_Replication.md"
OUTPUT = ROOT / "Final_Paper_EN_Chinese_Replication.docx"
FONT = "Times New Roman"
SIZE = Pt(12)
LINE_SPACING = 1.5
BODY_WIDTH = Inches(6.5)
FIGURE_WIDTH = Inches(6.0)


# --------------------------------------------------------------------------- #
# Markdown -> intermediate docx
# --------------------------------------------------------------------------- #

def markdown_to_docx(md_text: str, target: Path) -> None:
    """Run pandoc, converting image links but leaving structure to post-processing."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as handle:
        handle.write(md_text)
        temp_md = Path(handle.name)
    try:
        subprocess.run(
            [
                "pandoc", str(temp_md), "-o", str(target),
                "--from", "markdown+pipe_tables+implicit_figures-raw_html",
                "--to", "docx",
                f"--resource-path={ROOT}",
            ],
            check=True, capture_output=True,
        )
    finally:
        temp_md.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# XML helpers
# --------------------------------------------------------------------------- #

def _set_page_number_header(section) -> None:
    """Place a plain page number in the header (APA 7)."""
    paragraph = section.header.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = "PAGE"
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    run._r.append(begin); run._r.append(instr); run._r.append(end)
    for r in paragraph.runs:
        r.font.name = FONT
        r.font.size = SIZE


def _force_font(document) -> None:
    """Set the document default font, including the complex/east-asian slots."""
    style = document.styles["Normal"]
    style.font.name = FONT
    style.font.size = SIZE
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts"); rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), FONT)


def _format_paragraph(paragraph, *, first_line: bool = True, spacing: float = LINE_SPACING) -> None:
    fmt = paragraph.paragraph_format
    fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    fmt.line_spacing = spacing
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)
    fmt.widow_control = True
    if first_line:
        fmt.first_line_indent = Inches(0.5)
    else:
        fmt.first_line_indent = Inches(0)
    for run in paragraph.runs:
        run.font.name = FONT
        run.font.size = SIZE


def _style_table(table) -> None:
    """APA tables: borders only above/below the table and under the header row."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Fixed layout driven by an explicit tblGrid.  Setting cell widths alone is
    # ignored when the grid disagrees, which lets autofit squeeze long headers
    # until they break mid-word.
    table.autofit = False
    widths = [Inches(1.15), Inches(0.45), Inches(0.5), Inches(0.5), Inches(0.45),
              Inches(0.68), Inches(0.80), Inches(0.72)]
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    for tag in ("w:tblLayout", "w:tblW"):
        for existing in tbl_pr.findall(qn(tag)):
            tbl_pr.remove(existing)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)
    tbl_w = OxmlElement("w:tblW")
    tbl_w.set(qn("w:type"), "dxa")
    tbl_w.set(qn("w:w"), str(int(sum(w.inches for w in widths) * 1440)))
    tbl_pr.append(tbl_w)
    for existing in tbl.findall(qn("w:tblGrid")):
        tbl.remove(existing)
    grid = OxmlElement("w:tblGrid")
    for width in widths:
        column = OxmlElement("w:gridCol")
        column.set(qn("w:w"), str(int(width.inches * 1440)))
        grid.append(column)
    tbl_pr.addnext(grid)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width
            tc_pr = cell._tc.get_or_add_tcPr()
            for existing in tc_pr.findall(qn("w:tcW")):
                tc_pr.remove(existing)
            tc_w = OxmlElement("w:tcW")
            tc_w.set(qn("w:type"), "dxa")
            tc_w.set(qn("w:w"), str(int(width.inches * 1440)))
            tc_pr.append(tc_w)

    # Repeat the header row when a table runs onto a second page.
    first_row = table.rows[0]
    tr_pr = first_row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)
    for existing in tbl_pr.findall(qn("w:tblBorders")):
        tbl_pr.remove(existing)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single"); element.set(qn("w:sz"), "8")
        element.set(qn("w:space"), "0"); element.set(qn("w:color"), "000000")
        borders.append(element)
    for edge in ("left", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "none"); element.set(qn("w:sz"), "0")
        element.set(qn("w:space"), "0"); element.set(qn("w:color"), "auto")
        borders.append(element)
    tbl_pr.append(borders)
    for row_index, row in enumerate(table.rows):
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                _format_paragraph(paragraph, first_line=False, spacing=1.0)
                paragraph.paragraph_format.space_before = Pt(2)
                paragraph.paragraph_format.space_after = Pt(2)
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    if row_index == 0:
                        run.font.bold = True
            if row_index == 0:
                tc_pr = cell._tc.get_or_add_tcPr()
                for existing in tc_pr.findall(qn("w:tcBorders")):
                    tc_pr.remove(existing)
                tc_borders = OxmlElement("w:tcBorders")
                bottom = OxmlElement("w:bottom")
                bottom.set(qn("w:val"), "single"); bottom.set(qn("w:sz"), "6")
                bottom.set(qn("w:space"), "0"); bottom.set(qn("w:color"), "000000")
                tc_borders.append(bottom)
                tc_pr.append(tc_borders)


# --------------------------------------------------------------------------- #
# Main build
# --------------------------------------------------------------------------- #

def build() -> Path:
    md = SOURCE.read_text(encoding="utf-8")

    # 1. Front matter: title, byline, abstract, keywords.
    title = re.search(r"(?m)^# (.+)$", md).group(1).strip()
    byline = re.search(r"(?m)^Lu Xin\s*\n(?:Mentor:.*)$", md).group(0)
    abstract = re.search(r"(?ms)^## Abstract\n\n(.*?)\n\n\*Keywords", md).group(1).strip()
    keywords = re.search(r"(?m)^\*Keywords:\*\s*(.+)$", md).group(1).strip()

    # 2. Body, references, and the table/figure blocks that APA moves to the end.
    body = md[md.index("## Related Work"):md.index("## References")]
    references_block = md[md.index("## References"):]

    table_blocks = re.findall(r"(?ms)^\*\*Table \d+\*\*.*?(?=\n\n\*\*Figure|\n\n### |\n\n## |\Z)", body)
    figure_blocks = re.findall(r"(?ms)^\*\*Figure \d+\*\*.*?(?=\n\n\*\*Figure|\n\n### |\n\n## |\Z)", body)
    body_no_floats = re.sub(r"(?ms)^\*\*(?:Table|Figure) \d+\*\*.*?(?=\n\n\*\*Figure|\n\n### |\n\n## |\Z)", "", body)

    display = "\n\n".join([
        f"# {title}",
        byline,
        "## Abstract",
        abstract,
        f"*Keywords:* {keywords}",
        body_no_floats.strip(),
        references_block.strip(),
        "# Tables",
        *[block.strip() for block in table_blocks],
        "# Figures",
        *[block.strip() for block in figure_blocks],
    ]) + "\n"

    markdown_to_docx(display, OUTPUT)

    document = docx.Document(OUTPUT)
    _force_font(document)

    section = document.sections[0]
    for attr in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, attr, Inches(1))
    _set_page_number_header(section)

    # Walk the document, applying APA structure.
    paragraphs = list(document.paragraphs)
    headings = {"# ": "title", "## ": "level1", "### ": "level2"}
    in_references = False
    first_body_paragraph_done = False

    # The abstract and keyword lines sit between the byline and the first heading.
    abstract_zone = False
    for paragraph in paragraphs:
        raw = paragraph.text.strip()
        style_name = paragraph.style.name if paragraph.style else ""

        if paragraph.style and style_name.startswith("Heading"):
            level = int(style_name.split()[-1]) if style_name.split()[-1].isdigit() else 1
            if raw.lower().startswith("references"):
                in_references = True
            paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            # APA 7 double-spaces the title page; body headings use 1.5.
            is_title = raw.split("\n")[0].strip() == title
            paragraph.paragraph_format.line_spacing = 2.0 if is_title else LINE_SPACING
            paragraph.paragraph_format.first_line_indent = Inches(0)
            paragraph.paragraph_format.space_before = Pt(12 if level > 1 else 0)
            paragraph.paragraph_format.space_after = Pt(0)
            for run in paragraph.runs:
                run.font.name = FONT
                run.font.size = SIZE
                run.font.bold = True
                run.font.color.rgb = RGBColor(0, 0, 0)
            if level >= 2:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            continue

        if not raw:
            continue

        # Title / byline / abstract / keywords.
        # Pandoc can merge the H1 title with the byline into one paragraph, so
        # match on the first line rather than exact equality.
        if raw.split("\n")[0].strip() == title:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _format_paragraph(paragraph, first_line=False, spacing=2.0)
            for run in paragraph.runs:
                run.font.bold = True
            continue
        if raw.startswith("Lu Xin"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _format_paragraph(paragraph, first_line=False, spacing=2.0)
            continue

        if in_references:
            _format_paragraph(paragraph, first_line=False)
            fmt = paragraph.paragraph_format
            fmt.left_indent = Inches(0.5)
            fmt.first_line_indent = Inches(-0.5)     # hanging indent
            fmt.space_after = Pt(6)
            continue

        # Table titles (italic, above the table) and figure captions.
        if re.match(r"^(Table|Figure) \d+$", raw) or raw.startswith(("*Note.", "Accuracy and ECE", "MLE M-ratio", "Bayesian M-ratio", "Accuracy on fictional")):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            _format_paragraph(paragraph, first_line=False)
            continue

        _format_paragraph(paragraph, first_line=True)

    for table in document.tables:
        _style_table(table)

    # APA 7 places tables and figures after the reference list, each on its own
    # page: start Tables/Figures sections fresh, then break before every item.
    body = document.element.body
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text in {"Tables", "Figures"}:
            paragraph.paragraph_format.page_break_before = True
        elif re.match(r"^(Table|Figure) \d+$", text):
            paragraph.paragraph_format.page_break_before = True
        elif text == "References":
            paragraph.paragraph_format.page_break_before = True

    # Scale images to the text column and centre them; each figure starts a page.
    seen_image = False
    for paragraph in document.paragraphs:
        if paragraph._p.findall(".//" + qn("w:drawing")):
            if seen_image:
                paragraph.paragraph_format.page_break_before = True
            seen_image = True
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for shape in document.inline_shapes:
        try:
            ratio = shape.width / shape.height
        except ZeroDivisionError:
            continue
        if ratio >= 1:
            shape.width = FIGURE_WIDTH
            shape.height = int(FIGURE_WIDTH / ratio)
        else:
            shape.height = Inches(4.5)
            shape.width = int(Inches(4.5) * ratio)

    document.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
    print(f"size: {path.stat().st_size:,} bytes")
    sys.exit(0)
