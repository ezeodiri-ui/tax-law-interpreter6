import re

STOP = set("""a an the of to in on for by with and or as at is are be been was were it its this that these those
from which who whom shall may any such not no nor than then there their them his her he she they we you i
under into upon where when whether if has have had do does did can could would should will also other
""".split())


def stem(w: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("\u20a6", " naira ")
    raw = re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*%?", text)
    out = []
    for t in raw:
        t = t.replace(",", "")
        if t in STOP or len(t) < 2 and not t.isdigit():
            continue
        out.append(stem(t))
    return out
