"""Query understanding helpers: abbreviations, legacy->new terminology, explicit citations."""
import re

ACT_ALIASES = {
    "NTAA": ["ntaa", "nigeria tax administration act", "tax administration act", "administration act"],
    "NTA": ["nta", "nigeria tax act", "tax act"],
    "NRSEA": ["nrsea", "nrs act", "nigeria revenue service (establishment) act", "nigeria revenue service act", "revenue service act", "nrs establishment act"],
    "JRBEA": ["jrbea", "jrb act", "joint revenue board act", "joint revenue board of nigeria (establishment) act"],
}

# Terms people still use (old laws/institutions) -> vocabulary of the 2025 Acts. Appended to the BM25 query.
EXPANSIONS = {
    r"\bfirs\b|federal inland revenue service": "Nigeria Revenue Service Service Executive Chairman",
    r"\bnrs\b": "Nigeria Revenue Service",
    r"\bjtb\b|joint tax board": "Joint Revenue Board",
    r"\bcita\b|companies income tax": "company income tax total profits",
    r"\bpita\b|personal income tax": "individual income tax chargeable income",
    r"\bcgt\b|capital gains tax": "chargeable gains disposal assets",
    r"\bwht\b|withholding tax": "deduction at source tax",
    r"\bpaye\b|pay as you earn": "Pay As You Earn employment income deduction",
    r"\bvat\b": "value added tax",
    r"\btat\b|tax appeal tribunal": "Tax Appeal Tribunal appeal",
    r"\btin\b|tax identification number|tax id": "Tax ID taxpayer identification",
    r"\bsme\b|small business": "small company turnover",
    r"\bvasp\b|crypto|digital asset": "virtual assets digital assets service providers",
    r"\bsep\b|significant economic presence": "significant economic presence non-resident",
    r"\bstamp dut": "stamp duties dutiable instruments",
    r"\btax holiday|pioneer status": "economic development incentive certificate priority sector tax credit",
    r"\bpenalt": "administrative penalty offence fine",
    r"\bomb?budsman|tax ombud": "Tax Ombud complaint",
    r"\bsue\b|\bsuing\b|lawsuit|legal action": "suit legal proceedings limitation of suits notice",
}

CITE_RE = re.compile(
    r"\b(?:section|sections|sec\.?|s\.)\s*(\d{1,3})(?:\s*\(([0-9a-z]+)\))?", re.I)
SCHED_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|nin(?:e)?th|tenth|eleventh|twelfth)\s+schedule\b", re.I)


def detect_acts(q: str) -> list[str]:
    ql = q.lower()
    found = []
    # longest alias first so 'tax administration act' is not read as 'tax act'
    for code, al in ACT_ALIASES.items():
        for a in sorted(al, key=len, reverse=True):
            if re.search(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", ql):
                found.append(code)
                break
    if "NTAA" in found and "NTA" in found:
        # 'tax administration act' contains 'tax act'-like tokens only via 'nta'; keep explicit mentions
        if not re.search(r"(?<![a-z])nta(?![a-z])|nigeria tax act", q.lower()):
            found.remove("NTA")
    return found


def explicit_citations(q: str) -> tuple[list[int], list[str]]:
    secs = [int(m.group(1)) for m in CITE_RE.finditer(q)]
    scheds = [m.group(1).title().replace("Nineth", "Ninth") for m in SCHED_RE.finditer(q)]
    return secs, scheds


def expand(q: str) -> str:
    extra = [v for pat, v in EXPANSIONS.items() if re.search(pat, q, re.I)]
    return q + " " + " ".join(extra) if extra else q


# Institutions are created by one Act; naming them is a strong hint about where the answer lives.
ACT_HINTS = [
    (r"revenue service|\bnrs\b|\bfirs\b|executive chairman|governing board of the (?:nigeria )?revenue|warrant of deduction|technical committee", "NRSEA"),
    (r"joint revenue board|\bjrb\b|joint tax board|tax ombud|tax appeal|\btribunal|\bcommissioner of the tribunal", "JRBEA"),
    (r"stamp dut|\bvat\b|value added tax|capital gains|development levy|economic development incentive|capital allowance|royalt|dutiable", "NTA"),
    (r"\bpenalt|\bobjection|\bassessment|tax id\b|\breturns?\b|books of account|offence|\brefund|\bauditing|\bcompliance", "NTAA"),
]
SCHEDULE_INTENT = re.compile(r"\b(rates?|bands?|table|list|schedule|royalt\w*|allowances?)\b", re.I)
DEFINITION_INTENT = re.compile(r"\b(defin\w+|meaning of|what is (?:a|an|the)\b|what does .* mean|means?)\b", re.I)


def act_hints(q: str) -> list[str]:
    return [a for pat, a in ACT_HINTS if re.search(pat, q, re.I)]
