"""
Parse an extracted Act into its legal hierarchy:
    Act -> Chapter -> Part -> Section -> Subsection -> Paragraph   (+ Schedules)

Strategy
  * The 'Arrangement of Sections' (table of contents) is the authoritative list of
    section numbers and titles.
  * Section starts in the body are located *sequentially* (section n+1 can only
    start after section n), which makes the parser immune to numbered lists that
    look like section numbers.
  * Chapter / Part headings are read from the body and inherited by the sections
    that follow them.
  * Schedules are detected by their standalone '<Ordinal> Schedule' heading.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from .pdf_extract import extract_document, build_vocab, dehyphenate

ACTS = {
    "NTA": dict(file="NIGERIA_TAX_ACT__2025.pdf", title="Nigeria Tax Act, 2025", act_no=7,
                aliases=["nta", "nigeria tax act", "tax act", "ntax act"]),
    "NTAA": dict(file="NIGERIA_TAX_ADMINISTRATION_ACT__2025.pdf", title="Nigeria Tax Administration Act, 2025", act_no=5,
                 aliases=["ntaa", "nigeria tax administration act", "tax administration act", "administration act"]),
    "NRSEA": dict(file="NIGERIA_REVENUE_SERVICE__ESTABLISHMENT__ACT__2025.pdf", title="Nigeria Revenue Service (Establishment) Act, 2025", act_no=4,
                  aliases=["nrsea", "nrs act", "nigeria revenue service act", "nigeria revenue service (establishment) act", "revenue service act"]),
    "JRBEA": dict(file="JOINT_REVENUE_BOARD_OF_NIGERIA__ESTABLISHMENT__ACT__2025.pdf", title="Joint Revenue Board of Nigeria (Establishment) Act, 2025", act_no=6,
                  aliases=["jrbea", "jrb act", "joint revenue board act", "joint revenue board of nigeria (establishment) act"]),
}

ORD = r"(first|second|third|fourth|fifth|sixth|seventh|eighth|nin(?:e)?th|tenth|eleventh|twelfth)"
SCHED_RE = re.compile(rf"^{ORD}\s+schedule$", re.I)
CHAPTER_RE = re.compile(r"^CHAPTER\s+([0-9]+|[A-Z]+)\b")
PART_RE = re.compile(r"^PART\s+([IVXL]+|[0-9]+)\b")
SUB_RE = re.compile(r"^\(([0-9]+[A-Za-z]?|[a-z]{1,4}|[A-Z]{1,2})\)\s")
DEF_RE = re.compile(r"^[\u201c\"]([^\u201d\"]{2,120})[\u201d\"]\s*(means|includes|has|refers|shall|is|are)?")
NUM_RE = re.compile(r"^(\d{1,3})\.\s+")
ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv"}


@dataclass
class Para:
    marker: str          # '(1)', '(a)', '3.' or ''
    level: int           # 1 numeric subsection, 2 letter, 3 roman, 0 none
    text: str
    kind: str = "text"   # text | table | def | heading
    page: int = 0
    gaz: str | None = None


@dataclass
class Section:
    act: str
    number: int
    title: str
    chapter: str
    part: str
    paras: list[Para] = field(default_factory=list)
    page_start: str | None = None
    page_end: str | None = None


@dataclass
class Schedule:
    act: str
    ordinal: str
    title: str
    refs: str
    paras: list[Para] = field(default_factory=list)
    page_start: str | None = None


def _is_caps_heading(t: str) -> bool:
    letters = [c for c in t if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.85


def _table_text(rows: list[list[str]], carried: list[str] | None = None):
    """Header-aware rendering: 'Col: value | Col: value' per row. Tables that continue on the next
    page repeat no header row, so the previous table's header is carried over (same column count)."""
    clean = [[re.sub(r"\bN(?=\d)", "\u20a6", c).strip() for c in r] for r in rows]
    hdr = clean[0]
    looks_hdr = all(h and not re.fullmatch(r"[\d.,%\u20a6 ]+", h) for h in hdr if h) and not re.match(r"^\d+$", hdr[0] or "x")
    if not looks_hdr and carried and len(carried) == len(hdr):
        hdr, looks_hdr, body = carried, True, clean
    else:
        body = clean[1:] if looks_hdr else clean
    out = []
    for r in body:
        if looks_hdr and len(r) == len(hdr):
            cells = [f"{h}: {c}" for h, c in zip(hdr, r) if c]
        else:
            cells = [c for c in r if c]
        if cells:
            out.append(" | ".join(cells))
    return "\n".join(out), (hdr if looks_hdr else carried)


def to_paras(items: list[dict]) -> list[Para]:
    """Merge physical lines into logical paragraphs keyed by their legal marker."""
    paras: list[Para] = []
    prev_alpha = ""
    carried = None
    for it in items:
        if it["kind"] == "table":
            txt, carried = _table_text(it["rows"], carried)
            for row in txt.split("\n"):
                paras.append(Para("", 0, row, "table", it["page"], it.get("gaz")))
            continue
        t = it["text"].strip()
        if not t:
            continue
        m = SUB_RE.match(t)
        if m:
            lab = m.group(1)
            if lab.isdigit() or re.fullmatch(r"\d+[A-Za-z]", lab):
                level = 1
            elif lab in ROMAN and not (len(lab) == 1 and prev_alpha and ord(prev_alpha) + 1 == ord(lab)):
                level = 3
            elif lab.isalpha():
                level = 2
                if len(lab) == 1:
                    prev_alpha = lab
            else:
                level = 2
            paras.append(Para(f"({lab})", level, t, "text", it["page"], it.get("gaz")))
            continue
        if DEF_RE.match(t):
            paras.append(Para("", 0, t, "def", it["page"], it.get("gaz")))
            continue
        if NUM_RE.match(t) and re.match(r"^\d{1,3}\.\s+[A-Z(\u201c\"]", t):
            paras.append(Para(NUM_RE.match(t).group(0).strip(), 1, t, "text", it["page"], it.get("gaz")))
            continue
        if paras and paras[-1].kind != "table" and not (_is_caps_heading(t) and len(t) < 90):
            paras[-1].text += " " + t
        else:
            paras.append(Para("", 0, t, "heading" if _is_caps_heading(t) else "text", it["page"], it.get("gaz")))
    for p in paras:
        p.text = re.sub(r"\s+", " ", p.text) if p.kind != "table" else p.text
    return paras


def _parse_toc(lines: list[str]) -> list[tuple[int, str]]:
    secs: list[tuple[int, str]] = []
    i = 0
    n = len(lines)
    while i < n:
        t = lines[i].strip()
        m = re.match(r"^(\d{1,3})\.\s*(.*)$", t)
        if m and not re.match(r"^20\d\d\b", t):
            num, title = int(m.group(1)), m.group(2).strip()
            j = i + 1
            if not title and j < n:
                title = lines[j].strip(); j += 1
            # title continuation lines
            while j < n and not re.match(r"^\d{1,3}\.", lines[j].strip()) \
                    and not re.match(r"^(PART|Part|CHAPTER|Chapter|SCHEDULES?)\b", lines[j].strip()):
                title += " " + lines[j].strip(); j += 1
            title = re.sub(r"\s+", " ", title)
            title = re.sub(r"\s+SCHEDULES?\b.*$", "", title, flags=re.I)
            secs.append((num, title))
            i = j
        else:
            i += 1
    return secs


def parse_act(code: str, pdf_path: str, log=print) -> tuple[list[Section], list[Schedule]]:
    blocks = extract_document(pdf_path)
    vocab = build_vocab(blocks)
    # dehyphenate within runs of consecutive line blocks
    run: list[dict] = []

    def flush():
        if run:
            fixed = dehyphenate([r["text"] for r in run], vocab)
            if len(fixed) == len(run):
                for r, f in zip(run, fixed):
                    r["text"] = f
            else:  # some lines merged: rebuild
                k = 0
                for f in fixed:
                    run[k]["text"] = f
                    k += 1
                del run[k:]
        run.clear()
    # dehyphenation that merges lines changes length; do it with index mapping
    items: list[dict] = []
    buf: list[dict] = []
    for b in blocks + [{"kind": "end"}]:
        if b["kind"] == "line":
            buf.append(b)
            continue
        if buf:
            texts = [x["text"] for x in buf]
            fixed = []
            k = 0
            while k < len(texts):
                cur = texts[k]
                if cur.endswith("-") and k + 1 < len(texts):
                    m1 = re.search(r"([A-Za-z]+)-$", cur)
                    m2 = re.match(r"([A-Za-z]+)", texts[k + 1])
                    if m1 and m2 and texts[k + 1][:1].islower() and (m1.group(1) + m2.group(1)).lower() in vocab:
                        fixed.append((buf[k], cur[:-1] + texts[k + 1])); k += 2; continue
                fixed.append((buf[k], cur)); k += 1
            for src, txt in fixed:
                items.append(dict(src, text=txt))
            buf = []
        if b["kind"] == "table":
            items.append(b)

    texts = [it.get("text", "") if it["kind"] == "line" else "\u0000TABLE" for it in items]

    # ---- table of contents -------------------------------------------------
    toc_start = next(i for i, t in enumerate(texts) if "ARRANGEMENT OF SECTIONS" in t.upper())
    toc_end = next(i for i in range(toc_start, len(texts)) if re.match(r"^ACT\s+NO", texts[i].strip(), re.I))
    toc = _parse_toc(texts[toc_start:toc_end])
    log(f"[{code}] TOC lists {len(toc)} sections")

    # ---- locate body section starts sequentially -------------------------
    starts: list[tuple[int, int, str]] = []  # (num, idx, title)
    cursor = toc_end + 1
    missing = []
    for num, title in toc:
        pat = re.compile(rf"^{num}\.\s*(?=[(A-Z\u201c\"])")
        found = None
        for i in range(cursor, len(items)):
            if items[i]["kind"] == "line" and pat.match(texts[i]):
                found = i
                break
        if found is None:
            missing.append(num)
            continue
        starts.append((num, found, title))
        cursor = found + 1
    if missing:
        log(f"[{code}] WARNING sections not located in body: {missing}")

    # ---- schedules start ---------------------------------------------------
    last_idx = starts[-1][1]
    sched_idx = [i for i in range(last_idx, len(items)) if items[i]["kind"] == "line" and SCHED_RE.match(texts[i].strip())]
    body_end = sched_idx[0] if sched_idx else len(items)
    # include a preceding 'SCHEDULES' divider in the schedule region, not in the last section
    if body_end > 0 and texts[body_end - 1].strip().upper() in ("SCHEDULES", "SCHEDULE"):
        body_end -= 1

    # ---- sections ----------------------------------------------------------
    sections: list[Section] = []
    chapter, part = "", ""
    bounds = [s[1] for s in starts] + [body_end]
    for k, (num, idx, title) in enumerate(starts):
        seg = items[idx:bounds[k + 1]]
        kept: list[dict] = []
        j = 0
        # headings: strip and update context
        while j < len(seg):
            it = seg[j]
            if it["kind"] == "line":
                t = it["text"].strip()
                if CHAPTER_RE.match(t) and j > 0 and _is_caps_heading(t):
                    h = t; j += 1
                    while j < len(seg) and seg[j]["kind"] == "line" and _is_caps_heading(seg[j]["text"]) \
                            and not PART_RE.match(seg[j]["text"]) and not CHAPTER_RE.match(seg[j]["text"]) and len(h) < 140:
                        h += " " + seg[j]["text"].strip(); j += 1
                    chapter, part = h, ""
                    continue
                if PART_RE.match(t) and j > 0 and _is_caps_heading(t):
                    h = t; j += 1
                    while j < len(seg) and seg[j]["kind"] == "line" and _is_caps_heading(seg[j]["text"]) \
                            and not PART_RE.match(seg[j]["text"]) and not CHAPTER_RE.match(seg[j]["text"]) and len(h) < 140:
                        h += " " + seg[j]["text"].strip(); j += 1
                    part = h
                    continue
            kept.append(it)
            j += 1
        # The heading that precedes section k+1 sits at the *tail* of section k's slice;
        # contexts were updated during the loop above, but the current section must use the
        # context in force *before* its own tail headings. Recompute by splitting tail.
        sec = Section(code, num, title, "", "")
        sec.page_start = kept[0].get("gaz") if kept else None
        sec.page_end = kept[-1].get("gaz") if kept else None
        # first line: strip the section number
        first = kept[0]
        first = dict(first, text=re.sub(r"^\d{1,3}\.\s*", "", first["text"], count=1))
        sec.paras = to_paras([first] + kept[1:])
        sections.append(sec)

    # Assign chapter/part with correct (pre-tail) context in a second pass
    chapter, part = "", ""
    pre = items[toc_end + 1:starts[0][1]]
    j = 0
    while j < len(pre):
        it = pre[j]
        if it["kind"] == "line" and _is_caps_heading(it["text"].strip()):
            t = it["text"].strip()
            if CHAPTER_RE.match(t) or PART_RE.match(t):
                h = t; j += 1
                while j < len(pre) and pre[j]["kind"] == "line" and _is_caps_heading(pre[j]["text"]) \
                        and not PART_RE.match(pre[j]["text"]) and not CHAPTER_RE.match(pre[j]["text"]) and len(h) < 140:
                    h += " " + pre[j]["text"].strip(); j += 1
                if CHAPTER_RE.match(t):
                    chapter, part = h, ""
                else:
                    part = h
                continue
        j += 1
    for k, (num, idx, title) in enumerate(starts):
        # headings located between start k and start k+1 apply to section k+1.
        sections[k].chapter, sections[k].part = chapter, part
        seg = items[idx:bounds[k + 1]]
        j = 0
        while j < len(seg):
            it = seg[j]
            if it["kind"] == "line" and j > 0 and _is_caps_heading(it["text"].strip()):
                t = it["text"].strip()
                if CHAPTER_RE.match(t) or PART_RE.match(t):
                    h = t; j += 1
                    while j < len(seg) and seg[j]["kind"] == "line" and _is_caps_heading(seg[j]["text"]) \
                            and not PART_RE.match(seg[j]["text"]) and not CHAPTER_RE.match(seg[j]["text"]) and len(h) < 140:
                        h += " " + seg[j]["text"].strip(); j += 1
                    if CHAPTER_RE.match(t):
                        chapter, part = h, ""
                    else:
                        part = h
                    continue
            j += 1
    # tidy heading strings
    for s in sections:
        s.chapter = re.sub(r"\s+", " ", s.chapter).strip()
        s.part = re.sub(r"\s+", " ", s.part).strip()

    # ---- schedules ---------------------------------------------------------
    schedules: list[Schedule] = []
    sb = sched_idx + [len(items)]
    for k, si in enumerate(sched_idx):
        seg = items[si:sb[k + 1]]
        # certificate / explanatory memorandum trails the last schedule
        cut = next((q for q, it in enumerate(seg) if it["kind"] == "line" and re.match(r"^(I certify|EXPLANATORY MEMORANDUM)", it["text"])), len(seg))
        seg = seg[:cut]
        ordinal = seg[0]["text"].strip().split()[0].title().replace("Nineth", "Ninth")
        j = 1
        refs, title = [], ""
        while j < len(seg) and seg[j]["kind"] == "line":
            t = seg[j]["text"].strip()
            if re.match(r"^(Sections?|Paragraphs?)\b", t) or (refs and not _is_caps_heading(t) and not NUM_RE.match(t) and not PART_RE.match(t) and len(refs) < 6):
                refs.append(t); j += 1; continue
            if _is_caps_heading(t) and not title and not PART_RE.match(t):
                title = t; j += 1
                while j < len(seg) and seg[j]["kind"] == "line" and _is_caps_heading(seg[j]["text"]) and not PART_RE.match(seg[j]["text"]) and len(title) < 100:
                    title += " " + seg[j]["text"].strip(); j += 1
                continue
            break
        # rotated landscape schedules print the title above the heading
        sc = Schedule(code, ordinal, re.sub(r"\s+", " ", title).strip(), " ".join(refs).strip(), to_paras(seg[j:]),
                      seg[0].get("gaz"))
        schedules.append(sc)
    # Landscape schedules print their caps title *above* the "<Ordinal> Schedule" line, so it was
    # captured as the tail of the previous schedule. Move it across.
    for a, b_ in zip(schedules, schedules[1:]):
        if not b_.title and a.paras and a.paras[-1].kind == "heading" and len(a.paras[-1].text) < 70:
            b_.title = a.paras.pop().text
    log(f"[{code}] parsed {len(sections)} sections, {len(schedules)} schedules")
    return sections, schedules
