#!/usr/bin/env python3
"""Generate docs/KITE_Method.pdf without external PDF dependencies."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import textwrap
from pathlib import Path


PAGE_W = 612
PAGE_H = 792
MARGIN_X = 54
MARGIN_TOP = 58
MARGIN_BOTTOM = 54
BODY_SIZE = 10
H1_SIZE = 20
H2_SIZE = 14
LINE_H = 13


def clean_inline(text: str) -> str:
    text = text.replace("`", "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return text


def wrap_text(text: str, width: int = 92) -> list[str]:
    if not text.strip():
        return [""]
    return textwrap.wrap(clean_inline(text), width=width, replace_whitespace=False) or [""]


def markdown_to_lines(markdown: str) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    in_code = False
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            lines.append(("code", line))
            continue
        if line.startswith("# "):
            lines.append(("h1", line[2:].strip()))
        elif line.startswith("## "):
            lines.append(("h2", line[3:].strip()))
        elif line.startswith("|"):
            cells = [clean_inline(cell.strip()) for cell in line.strip("|").split("|")]
            if cells and not all(set(cell) <= {"-", ":"} for cell in cells):
                lines.append(("table", " | ".join(cells)))
        elif line.startswith("- "):
            for idx, wrapped in enumerate(wrap_text(line[2:], width=88)):
                prefix = "- " if idx == 0 else "  "
                lines.append(("body", prefix + wrapped))
        elif re.match(r"^\d+\. ", line):
            lines.append(("body", clean_inline(line)))
        elif line.startswith("   "):
            for wrapped in wrap_text(line.strip(), width=88):
                lines.append(("body", "  " + wrapped))
        elif not line:
            lines.append(("blank", ""))
        else:
            for wrapped in wrap_text(line, width=92):
                lines.append(("body", wrapped))
    return lines


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def add_text(stream: list[str], x: float, y: float, text: str, font: str = "F1", size: int = BODY_SIZE) -> None:
    stream.append(f"BT /{font} {size} Tf {x:.1f} {y:.1f} Td ({pdf_escape(text)}) Tj ET")


def paginate(lines: list[tuple[str, str]]) -> list[list[str]]:
    pages: list[list[str]] = []
    stream: list[str] = []
    y = PAGE_H - MARGIN_TOP
    page_number = 1

    def new_page() -> None:
        nonlocal stream, y, page_number
        add_text(stream, MARGIN_X, 28, f"KITE Method | {page_number}", "F1", 8)
        pages.append(stream)
        page_number += 1
        stream = []
        y = PAGE_H - MARGIN_TOP

    for kind, text in lines:
        if kind == "blank":
            y -= LINE_H * 0.7
            continue
        if kind == "h1":
            needed = 34
        elif kind == "h2":
            needed = 26
        else:
            needed = LINE_H
        if y - needed < MARGIN_BOTTOM:
            new_page()

        if kind == "h1":
            add_text(stream, MARGIN_X, y, text, "F2", H1_SIZE)
            y -= 30
        elif kind == "h2":
            y -= 5
            add_text(stream, MARGIN_X, y, text, "F2", H2_SIZE)
            y -= 20
        elif kind == "code":
            add_text(stream, MARGIN_X + 14, y, text[:100], "F3", 8)
            y -= 11
        elif kind == "table":
            add_text(stream, MARGIN_X, y, text[:105], "F3", 8)
            y -= 11
        else:
            add_text(stream, MARGIN_X, y, text, "F1", BODY_SIZE)
            y -= LINE_H

    add_text(stream, MARGIN_X, 28, f"KITE Method | {page_number}", "F1", 8)
    pages.append(stream)
    return pages


def build_pdf(page_streams: list[list[str]], title: str) -> bytes:
    objects: list[bytes] = []

    def obj(data: str | bytes) -> int:
        if isinstance(data, str):
            data = data.encode("latin-1", errors="replace")
        objects.append(data)
        return len(objects)

    catalog_id = obj("PLACEHOLDER")
    pages_id = obj("PLACEHOLDER")
    font_regular_id = obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold_id = obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    font_mono_id = obj("<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
    page_ids = []

    for stream_lines in page_streams:
        stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
        content_id = obj(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        page_id = obj(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R /F3 {font_mono_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        )
        page_ids.append(page_id)

    objects[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode()
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()

    info_id = obj(
        f"<< /Title ({pdf_escape(title)}) /Author (KITE) "
        f"/CreationDate (D:{dt.datetime.now():%Y%m%d%H%M%S}) >>"
    )

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, data in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode())
        out.extend(data)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R /Info {info_id} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("docs/KITE_Method.md"))
    parser.add_argument("--output", type=Path, default=Path("docs/KITE_Method.pdf"))
    args = parser.parse_args()

    markdown = args.input.read_text()
    pages = paginate(markdown_to_lines(markdown))
    pdf = build_pdf(pages, "KITE Method")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pdf)
    print(f"wrote {args.output} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
