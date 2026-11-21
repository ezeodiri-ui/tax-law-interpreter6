"""
Layout-aware extraction of Nigerian Gazette PDFs.

Problems this module solves (all present in the four 2025 Acts):
  1. Marginal notes (side headings) sit beside the body text and, with naive
     extraction, are interleaved into sentences. We detect them by font size
     and horizontal position and remove them.
  2. Running page headers ("Nigeria Tax Act, 2025  2025 No. 7  A 395") are
     removed, but the Gazette page number is kept so every chunk can be cited
     to its printed page.
  3. Schedule tables (some printed rotated 90 degrees) are reconstructed into
     proper rows instead of a soup of words.
  4. Soft hyphenation at line ends ("cat-\\negories") is repaired.
"""
from __future__ import annotations
import re
import collections
import warnings
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning)
import fitz  # PyMuPDF

GAZETTE_RE = re.compile(r"\bA\s?(\d{3})\b")


def is_header(text: str) -> bool:
    """True for running page headers such as 'A 266 2025 No. 5' or
    'Nigeria Tax Act, 2025 2025 No. 7 A 395' (in any order)."""
    t = re.sub(r"\bA\s?\d{3}\b", "", text)
    t = re.sub(r"\b20\d\d\s+No\.?\s*\d+", "", t)
    t = re.sub(r"Joint Revenue Board of Nigeria", "", t)
    t = re.sub(r"[A-Za-z() ,\-]+Act,?\s*20\d\d", "", t)
    return t.strip() == "" and len(text) < 70



# The Gazette prints the naira sign as a glyph with no Unicode mapping, so the text
# layer shows only a wide gap before the amount ("penalty of     5,000,000"). Restore it,
# otherwise every monetary threshold and penalty loses its currency.
_NAIRA_GAP = re.compile(r"(?<=\S) {3,}(?=\d{1,3}(?:,\d{3})+(?:\.\d+)?\b)")


def restore_naira(raw: str) -> str:
    raw = _NAIRA_GAP.sub(" \u20a6", raw)
    return re.sub(r"\bN(?=\d{1,3}(?:,\d{3})+)", "\u20a6", raw)


def _norm(s: str) -> str:
    s = s.replace("\u00ad", "").replace("\u00a0", " ")
    return re.sub(r"[ \t]+", " ", s).strip()


def _percentile(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    return vals[min(len(vals) - 1, max(0, int(q * (len(vals) - 1))))]


def _page_lines(page) -> list[dict]:
    out = []
    d = page.get_text("dict")
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b["lines"]:
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = _norm(restore_naira("".join(s["text"] for s in ln["spans"])))
            # the struck-through N glyph is also emitted as a stray capital N at the end of the line
            text = re.sub(r"\s+N$", "", text)
            if text == "N":
                continue
            if not text:
                continue
            x0, y0, x1, y1 = ln["bbox"]
            out.append({
                "text": text,
                "size": max(s["size"] for s in spans),
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "dir": tuple(round(v) for v in ln["dir"]),
            })
    return out


def _clip_text(page, bbox) -> str:
    t = page.get_text("text", clip=fitz.Rect(bbox))
    return _norm(re.sub(r"\s*\n\s*", " ", t))


def _table_matrix(page, tab, rotated: int) -> list[list[str]]:
    """Return a row-major matrix of cell text, un-rotating if needed."""
    grid = []
    for row in tab.rows:
        grid.append([_clip_text(page, c) if c else "" for c in row.cells])
    if not grid:
        return []
    if rotated == 0:
        return grid
    # Rotated page: PyMuPDF sees original columns as rows. Transpose, and for
    # text running bottom-to-top (dir 0,-1) the original column order is reversed.
    cols = len(grid[0])
    mat = [[grid[r][c] for r in range(len(grid))] for c in range(cols)]
    if rotated == -1:
        mat = [list(reversed(r)) for r in mat]
        # rows[] came out top-to-bottom = original columns right-to-left, so
        # after transposing we reversed each row; row order is already correct.
    return mat


def extract_document(pdf_path: str) -> list[dict]:
    """Return reading-order blocks: {'kind': 'line'|'table', 'page': n, 'gaz': 'A 395', ...}"""
    doc = fitz.open(pdf_path)
    blocks: list[dict] = []
    for pno in range(len(doc)):
        page = doc[pno]
        lines = _page_lines(page)
        if not lines:
            continue

        # Gazette page number + running-header detection (header sits in the top ~125pt)
        gaz = None
        header_ids = set()
        for idx, ln in enumerate(lines):
            upright = ln["dir"] == (1, 0)
            if len(ln["text"]) < 70 and ((upright and ln["y0"] < 125) or not upright):
                t = ln["text"]
                if is_header(t):
                    header_ids.add(idx)
                    m = GAZETTE_RE.search(t)
                    if m:
                        gaz = f"A {m.group(1)}"
        for idx in header_ids:
            lines[idx]["_hdr"] = True

        rot_lines = [l for l in lines if l["dir"] != (1, 0)]
        up_lines = [l for l in lines if l["dir"] == (1, 0)]
        rotated = 0
        if len(rot_lines) > len(up_lines):
            rotated = -1 if collections.Counter(l["dir"] for l in rot_lines).most_common(1)[0][0] == (0, -1) else 1

        # ---- tables -------------------------------------------------------
        table_boxes: list[tuple] = []
        tables_out: list[dict] = []
        try:
            found = page.find_tables()
            for tab in found.tables:
                mat = _table_matrix(page, tab, rotated)
                mat = [[c for c in r] for r in mat if any(c.strip() for c in r)]
                flat = " ".join(c for r in mat for c in r if c.strip())
                if is_header(flat) or len(flat) < 12:
                    continue
                ncols = max((len(r) for r in mat), default=0)
                cells = [c for r in mat for c in r]
                fill = sum(1 for c in cells if c.strip()) / max(1, len(cells))
                first_row = " ".join(c for c in (mat[0] if mat else []) if c.strip())
                # PyMuPDF reads the rule under the running header as a page-sized table: reject those
                bogus = is_header(first_row) or (ncols < 3 and fill < 0.6) or \
                    (ncols <= 2 and (tab.bbox[3] - tab.bbox[1]) > 0.6 * page.rect.height)
                if len(mat) >= 2 and ncols >= 2 and not bogus:
                    table_boxes.append(tuple(tab.bbox))
                    tables_out.append({"kind": "table", "page": pno + 1, "gaz": gaz,
                                       "y0": tab.bbox[1], "rows": mat, "rotated": bool(rotated)})
        except Exception:
            pass

        def in_table(l):
            cx, cy = (l["x0"] + l["x1"]) / 2, (l["y0"] + l["y1"]) / 2
            return any(b[0] - 2 <= cx <= b[2] + 2 and b[1] - 2 <= cy <= b[3] + 2 for b in table_boxes)

        # ---- body column & marginal notes --------------------------------
        body = [l for l in up_lines if l["size"] >= 10.4 and l["y0"] > 55 and not in_table(l)]
        if len(body) >= 8:  # pages made of 9pt text (arrangement of sections) have no marginal notes
            left = _percentile([l["x0"] for l in body], 0.04)
            right = _percentile([l["x1"] for l in body], 0.96)
        else:
            body = []
            left, right = 0, 10_000

        kept = []
        for l in up_lines:
            if l.get("_hdr"):
                continue  # running header
            if in_table(l):
                continue
            if l["size"] <= 9.4 and body and (l["x1"] <= left + 2 or l["x0"] >= right - 2):
                continue  # marginal note
            if l["size"] < 6:
                continue  # glyph debris
            kept.append(l)

        if rotated:
            kept = []
            for l in rot_lines:
                if l.get("_hdr") or in_table(l) or l["size"] < 6 or len(l["text"]) < 3:
                    continue
                if l["size"] <= 9.4:
                    continue  # rotated marginal notes / header remnants
                # rotated reading order: lines stack along x; map to pseudo y/x
                kept.append(dict(l, y0=l["x0"] if rotated == -1 else -l["x0"], x0=-l["y1"] if rotated == -1 else l["y0"]))
            kept += [l for l in up_lines if not l.get("_hdr") and not in_table(l) and l["size"] >= 6 and l["y0"] > 55 and l["size"] > 9.4]
        items = [{"kind": "line", "page": pno + 1, "gaz": gaz, "y0": l["y0"], "x0": l["x0"],
                  "size": l["size"], "text": l["text"]} for l in kept] + tables_out
        items.sort(key=lambda it: (round(it["y0"] / 3), it.get("x0", 0)))
        blocks.extend(items)
    return blocks


def build_vocab(blocks: list[dict]) -> set[str]:
    v = set()
    for b in blocks:
        if b["kind"] == "line":
            for w in re.findall(r"[A-Za-z]+", b["text"]):
                v.add(w.lower())
    return v


def dehyphenate(lines: list[str], vocab: set[str]) -> list[str]:
    """Join 'cat-' + 'egories' when the joined word exists elsewhere in the corpus
    unhyphenated; keep genuine hyphens ('non-resident')."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if cur.endswith("-") and i + 1 < len(lines):
            nxt = lines[i + 1]
            m1 = re.search(r"([A-Za-z]+)-$", cur)
            m2 = re.match(r"([A-Za-z]+)(.*)$", nxt)
            if m1 and m2 and nxt[:1].islower():
                joined = (m1.group(1) + m2.group(1)).lower()
                if joined in vocab:
                    cur = cur[:-1] + nxt
                    i += 2
                    out.append(cur)
                    continue
        out.append(cur)
        i += 1
    return out
