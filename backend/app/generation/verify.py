"""Deterministic post-checks on a generated answer (no second LLM call)."""
import re


def _norm(s: str) -> str:
    s = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"[^a-z0-9\u20a6%., ]+", " ", s.lower()).replace("  ", " ")


def verify(answer: str, sources: list[dict]) -> dict:
    n_sources = len(sources)
    cited = sorted({int(x) for x in re.findall(r"\[(\d{1,2})\]", answer)})
    bad_cites = [c for c in cited if c < 1 or c > n_sources]
    corpus = _norm(" ".join(s["text"] for s in sources))
    corpus_compact = re.sub(r"\s+", " ", corpus)
    unverified_quotes = []
    for q in re.findall(r"[\u201c\"]([^\u201d\"]{12,300})[\u201d\"]", answer):
        qn = re.sub(r"\s+", " ", _norm(q)).strip()
        if qn and qn not in corpus_compact:
            unverified_quotes.append(q)
    figs = set(re.findall(r"\u20a6\s?\d[\d,]*(?:\.\d+)?\s?(?:m|b|million|billion|trillion)?|\d+(?:\.\d+)?\s?%", answer))
    unsupported = []
    for f in figs:
        fn = re.sub(r"\s+", "", f.lower())
        if fn not in re.sub(r"\s+", "", corpus):
            unsupported.append(f)
    ok = not bad_cites and not unverified_quotes and not unsupported and bool(cited)
    return {"ok": ok, "citations_used": cited, "invalid_citations": bad_cites,
            "unverified_quotes": unverified_quotes, "unsupported_figures": unsupported,
            "uncited": not cited}
