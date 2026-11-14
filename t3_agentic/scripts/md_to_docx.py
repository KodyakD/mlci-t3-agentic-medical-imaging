"""Convert the T3 markdown report to a Word document.

Supports: headings, paragraphs, bullet lists, tables, fenced code, inline
bold/code, and ![image](path) figures.
"""
import os
import re
import sys

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = r"C:\Users\PC\Desktop\M2S1\MLCI"
SRC = os.path.join(ROOT, "docs", "Report_T3_Agentic_System.md")
DST = os.path.join(ROOT, "docs", "Report_T3_Agentic_System.docx")


def add_runs(par, text):
    for part in re.split(r"(\*\*.*?\*\*|`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = par.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1]); r.font.name = "Consolas"
            r.font.size = Pt(9)
        else:
            par.add_run(part)


def convert(md_path, out_path):
    doc = docx.Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    lines = open(md_path, encoding="utf-8").read().splitlines()
    i = 0
    in_code = False
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            p = doc.add_paragraph()
            r = p.add_run(line)
            r.font.name = "Consolas"
            r.font.size = Pt(8.5)
            i += 1
            continue
        if line.startswith("### "):
            doc.add_heading(line[4:], level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.startswith("# "):
            doc.add_heading(line[2:], level=1)
        elif re.match(r"^\s*[-*] ", line):
            text = re.sub(r"^\s*[-*] ", "", line)
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, text)
        elif line.strip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block]
            rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
            if rows:
                ncols = max(len(r) for r in rows)
                table = doc.add_table(rows=0, cols=ncols)
                table.style = "Light Grid Accent 1"
                for ri, r in enumerate(rows):
                    cells = table.add_row().cells
                    for ci in range(ncols):
                        txt = r[ci] if ci < len(r) else ""
                        par = cells[ci].paragraphs[0]
                        add_runs(par, txt)
                        if ri == 0:
                            for run in par.runs:
                                run.bold = True
            continue
        else:
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
            if m:
                path = m.group(2)
                if not os.path.isabs(path):
                    path = os.path.join(ROOT, path)
                if os.path.exists(path):
                    doc.add_picture(path, width=Inches(5.8))
                    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    cap = doc.add_paragraph(m.group(1))
                    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in cap.runs:
                        r.italic = True
                        r.font.size = Pt(9)
                        r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
                else:
                    doc.add_paragraph(f"[figure manquante: {path}]")
            elif line.strip():
                p = doc.add_paragraph()
                add_runs(p, line.strip())
        i += 1

    doc.save(out_path)
    return out_path


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    dst = sys.argv[2] if len(sys.argv) > 2 else DST
    print("written:", convert(src, dst))
