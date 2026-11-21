"""
Structure-aware legal chunking.

Principles (why not fixed-size windows):
  * The unit of law is the *section*. We never cut a sentence, and never separate a
    paragraph "(a)" from the lead-in words that give it meaning ("...is exempt where:").
  * Short sections stay whole. Long sections split on subsection boundaries "(1)", "(2)";
    a subsection that is itself too long splits on paragraph boundaries, and the lead-in
    clause is repeated in every piece ("chapeau carry").
  * Definitions are indexed one per chunk, so "what is a small company?" retrieves the
    definition itself rather than a 40-definition blob.
  * Schedule tables are indexed one *row* per chunk with column labels attached
    ("Name of Instruments: ... | New Rate: 0.375%"), so rates are never detached from
    the instrument they apply to.
  * Every chunk carries a context header (Act > Chapter > Part > Section title) in the
    text that is embedded/BM25-indexed, and stores prev/next links and the Gazette page.
  * Cross-references ("section 51 of the Nigeria Tax Administration Act") become graph
    edges, used at query time to pull in the provisions a rule depends on.
"""
from __future__ import annotations
import re
from .structure import ACTS, Section, Schedule, Para

TARGET_CHARS = 1900      # ~475 tokens: pack subsections up to this
MAX_CHARS = 3000         # a single subsection above this is split on paragraphs
WHOLE_SECTION_CHARS = 2600

ORDINALS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth",
            "Eleventh", "Twelfth", "Thirteenth"]
ORD_RE = r"(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Nin(?:e)?th|Tenth|Eleventh|Twelfth|Thirteenth|Thirtenth)"

ACT_NAME_PATTERNS = [
    ("NTAA", r"Nigeria\s+Tax\s+Administration\s+Act"),
    ("NTA", r"Nigeria\s+Tax\s+Act"),
    ("NRSEA", r"Nigeria\s+Revenue\s+Service\s*\(Establishment\)\s+Act|Nigeria\s+Revenue\s+Service\s+Act"),
    ("JRBEA", r"Joint\s+Revenue\s+Board(?:\s+of\s+Nigeria)?\s*(?:\(Establishment\)\s*)?Act"),
]


def _est(s: str) -> int:
    return len(s) // 4


def _join(paras: list[Para]) -> str:
    return "\n".join(p.text for p in paras)


def extract_refs(text: str, own_act: str) -> list[str]:
    """Return cross-reference ids like 'NTAA:s51' or 'NTA:sched:Fifth'."""
    refs: list[str] = []
    for m in re.finditer(r"\bsections?\s+((?:\d{1,3}(?:\s*\([0-9A-Za-z]+\))*\s*(?:,|and|or|to|–|-)?\s*)+)", text):
        window = text[m.end(): m.end() + 120]
        act = own_act
        for code, pat in ACT_NAME_PATTERNS:
            if re.match(r"\s*(?:of\s+(?:the\s+)?)?" + pat, window, re.I):
                act = code
                break
        for n in re.findall(r"(?<![\w(])(\d{1,3})(?![\d,])", re.sub(r"\([0-9A-Za-z]+\)", " ", m.group(1))):
            refs.append(f"{act}:s{int(n)}")
    for m in re.finditer(ORD_RE + r"\s+Schedule", text):
        window = text[m.end(): m.end() + 100]
        act = own_act
        for code, pat in ACT_NAME_PATTERNS:
            if re.match(r"\s*(?:to\s+|of\s+)?(?:this\s+Act\s+and\s+)?(?:the\s+)?" + pat, window, re.I):
                act = code
                break
        o = m.group(1).replace("Nineth", "Ninth").replace("Thirtenth", "Thirteenth")
        refs.append(f"{act}:sched:{o}")
    seen, out = set(), []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _chunk_obj(**kw) -> dict:
    return kw


def _header(act: str, sec: Section | None = None, sched: Schedule | None = None, extra: str = "") -> str:
    title = ACTS[act]["title"]
    if sec:
        bits = [title, sec.chapter, sec.part, f"Section {sec.number}: {sec.title}"]
    else:
        bits = [title, f"{sched.ordinal} Schedule: {sched.title}".strip(": "), sched.refs]
    if extra:
        bits.append(extra)
    return " > ".join(b for b in bits if b)


def _group_subsections(paras: list[Para]) -> list[list[Para]]:
    """Group paragraphs under their top-level subsection '(n)'."""
    groups: list[list[Para]] = []
    for p in paras:
        if p.level == 1 and p.marker.startswith("(") or not groups:
            groups.append([p])
        else:
            groups[-1].append(p)
    return groups


def _split_long_group(group: list[Para]) -> list[list[Para]]:
    """Split a long subsection on paragraph boundaries, carrying the lead-in."""
    lead = group[0]
    rest = group[1:]
    if not rest:
        return [group]
    pieces, cur, size = [], [], len(lead.text)
    for p in rest:
        if cur and size + len(p.text) > TARGET_CHARS:
            pieces.append([lead] + cur)
            cur, size = [], len(lead.text)
        cur.append(p)
        size += len(p.text)
    if cur:
        pieces.append([lead] + cur)
    # the lead appears in every piece except as given; mark carried lead for display
    return pieces


def chunk_section(sec: Section) -> list[dict]:
    act = sec.act
    base_id = f"{act}-s{sec.number}"
    label = f"s.{sec.number}"
    out: list[dict] = []
    paras = [p for p in sec.paras if p.text.strip()]
    full = _join(paras)

    def make(idx: int | None, ps: list[Para], extra: str = "", kind: str = "section", defterm: str | None = None):
        text = _join(ps)
        cid = base_id if idx is None else f"{base_id}-p{idx}"
        if defterm:
            cid = f"{base_id}-d{idx}"
        refs = extract_refs(text, act)
        hdr = _header(act, sec=sec, extra=extra)
        return _chunk_obj(
            id=cid, act=act, act_title=ACTS[act]["title"], kind=kind, section=sec.number, schedule=None,
            title=sec.title, chapter=sec.chapter, part=sec.part, label=label, sub=extra, term=defterm,
            text=text, embed_text=f"{hdr}\n{text}", gaz=ps[0].gaz or sec.page_start, gaz_end=ps[-1].gaz or sec.page_end,
            tokens=_est(text), refs=refs,
        )

    # ---- definitions sections: one chunk per defined term ------------------
    if re.search(r"interpretation|definitions", sec.title, re.I) and sum(p.kind == "def" for p in paras) >= 2:
        intro = [p for p in paras if p.kind != "def"][:0]
        cur: list[Para] = []
        defs: list[list[Para]] = []
        lead_in: list[Para] = []
        for p in paras:
            if p.kind == "def":
                if cur:
                    defs.append(cur)
                cur = [p]
            elif cur:
                cur.append(p)
            else:
                lead_in.append(p)
        if cur:
            defs.append(cur)
        for i, d in enumerate(defs, 1):
            term = re.match(r"^[\u201c\"]([^\u201d\"]+)", d[0].text).group(1).strip()
            out.append(make(i, d, extra=f"Definition of \u201c{term}\u201d", kind="definition", defterm=term))
        return _link(out)

    if len(full) <= WHOLE_SECTION_CHARS:
        out.append(make(None, paras))
        return out

    groups = _group_subsections(paras)
    units: list[tuple[list[Para], str]] = []
    for g in groups:
        if sum(len(p.text) for p in g) > MAX_CHARS and len(g) > 2:
            for piece in _split_long_group(g):
                units.append((piece, g[0].marker))
        else:
            units.append((g, g[0].marker))

    packed: list[list[Para]] = []
    cur, size = [], 0
    for ps, _ in units:
        sz = sum(len(p.text) for p in ps)
        if cur and size + sz > TARGET_CHARS:
            packed.append(cur)
            cur, size = [], 0
        cur.extend(ps)
        size += sz
    if cur:
        packed.append(cur)

    for i, ps in enumerate(packed, 1):
        markers = [p.marker for p in ps if p.level == 1 and p.marker.startswith("(")]
        rng = ""
        if markers:
            rng = f"Subsection{'s' if len(set(markers)) > 1 else ''} {markers[0]}" + (f"\u2013{markers[-1]}" if len(set(markers)) > 1 else "")
        out.append(make(i, ps, extra=rng))
    return _link(out)


def _link(chunks: list[dict]) -> list[dict]:
    for i, c in enumerate(chunks):
        c["prev"] = chunks[i - 1]["id"] if i else None
        c["next"] = chunks[i + 1]["id"] if i + 1 < len(chunks) else None
    return chunks


def chunk_schedule(sc: Schedule) -> list[dict]:
    act = sc.act
    base = f"{act}-sch{sc.ordinal}"
    out: list[dict] = []
    part = ""
    buf: list[Para] = []
    size = 0
    n = 0
    table_rows: list[Para] = []

    def emit(ps: list[Para], extra: str = "", kind: str = "schedule"):
        nonlocal n
        if not ps:
            return
        n += 1
        text = _join(ps)
        hdr = _header(act, sched=sc, extra=" > ".join(x for x in (part, extra) if x))
        out.append(_chunk_obj(
            id=f"{base}-{n}", act=act, act_title=ACTS[act]["title"], kind=kind, section=None, schedule=sc.ordinal,
            title=sc.title or f"{sc.ordinal} Schedule", chapter="", part=part, label=f"{sc.ordinal} Schedule", sub=extra,
            term=None, text=text, embed_text=f"{hdr}\n{text}", gaz=ps[0].gaz or sc.page_start, gaz_end=ps[-1].gaz,
            tokens=_est(text), refs=extract_refs(text + " " + sc.refs, act),
            schedule_refs=sc.refs,
        ))

    def flush_text():
        nonlocal buf, size
        emit(buf)
        buf, size = [], 0

    def flush_table():
        nonlocal table_rows
        # pack table rows into chunks of ~ 6 rows / 1500 chars; each row keeps its column labels
        cur, sz = [], 0
        for r in table_rows:
            if cur and (sz + len(r.text) > 1500 or len(cur) >= 4):
                emit(cur, kind="schedule_table")
                cur, sz = [], 0
            cur.append(r)
            sz += len(r.text)
        emit(cur, kind="schedule_table")
        table_rows = []

    for p in sc.paras:
        if p.kind == "table":
            if buf:
                flush_text()
            table_rows.append(p)
            continue
        if table_rows:
            flush_table()
        if p.kind == "heading" and re.match(r"^PART\b", p.text):
            flush_text()
            part = p.text
            continue
        starts_new = bool(re.match(r"^\d{1,3}\.", p.marker)) and p.level == 1
        if buf and (size + len(p.text) > TARGET_CHARS or (starts_new and size > 900)):
            flush_text()
        buf.append(p)
        size += len(p.text)
    if table_rows:
        flush_table()
    flush_text()
    return _link(out)


def extract_definitions(chunks: list[dict]) -> dict[str, list[dict]]:
    defs: dict[str, list[dict]] = {}
    for c in chunks:
        if c["kind"] == "definition":
            defs.setdefault(c["term"].lower(), []).append({"id": c["id"], "act": c["act"], "term": c["term"]})
    return defs


def build_chunks(parsed: dict) -> list[dict]:
    chunks: list[dict] = []
    for code, (secs, scheds) in parsed.items():
        for s in secs:
            chunks.extend(chunk_section(s))
        for sc in scheds:
            chunks.extend(chunk_schedule(sc))
        # definitions inside schedules (quoted terms) are kept as normal schedule text
    # drop references to things that do not exist
    ids = {c["id"] for c in chunks}
    sec_ids = {f"{c['act']}:s{c['section']}" for c in chunks if c["section"]}
    sch_ids = {f"{c['act']}:sched:{c['schedule']}" for c in chunks if c["schedule"]}
    for c in chunks:
        c["refs"] = [r for r in c["refs"] if r in sec_ids or r in sch_ids]
        own = f"{c['act']}:s{c['section']}" if c["section"] else None
        c["refs"] = [r for r in c["refs"] if r != own]
    return chunks
