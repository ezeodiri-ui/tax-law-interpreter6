"""
Hybrid retrieval engine.

  query -> analysis (acts, explicit citations, glossary expansion)
        -> BM25 over context-headed text      (exact legal vocabulary, numbers, section titles)
        -> dense vectors                       (paraphrase / concept match)
        -> Reciprocal Rank Fusion              (robust, no score calibration needed)
        -> citation injection                  ("section 148", "Ninth Schedule" are pinned to the top)
        -> optional cross-encoder rerank
        -> context assembly: stitch sibling chunks, follow cross-references, attach definitions
"""
from __future__ import annotations
import json
import pickle
import re
import numpy as np
from ..config import INDEX_DIR, DENSE_BACKEND, EMBED_MODEL, VOYAGE_MODEL, RERANK_MODEL
from .bm25 import BM25
from .text import tokenize
from .glossary import detect_acts, explicit_citations, expand, act_hints, SCHEDULE_INTENT, DEFINITION_INTENT
from .dense import make_encoder, LSAEncoder

GENERIC_TERMS = {"person", "tax", "service", "board", "minister", "government", "act", "year", "company", "income", "profit", "profits"}


class Engine:
    def __init__(self, index_dir=INDEX_DIR, backend=DENSE_BACKEND):
        self.chunks = json.loads((index_dir / "chunks.json").read_text())
        self.toc = json.loads((index_dir / "toc.json").read_text())
        self.defs = json.loads((index_dir / "definitions.json").read_text())
        self.by_id = {c["id"]: c for c in self.chunks}
        self.by_sec: dict[str, list[dict]] = {}
        for c in self.chunks:
            key = f"{c['act']}:s{c['section']}" if c["section"] else f"{c['act']}:sched:{c['schedule']}"
            self.by_sec.setdefault(key, []).append(c)
        self.bm25 = BM25([tokenize(c["embed_text"]) for c in self.chunks])
        # title/label field: matches questions phrased like a section heading
        self.bm25_title = BM25([tokenize(f"{c['title']} {c['part']} {c['chapter']} {c.get('term') or ''}") for c in self.chunks])
        self.emb = None
        self.enc = None
        p = index_dir / "embeddings.npy"
        if backend != "none" and p.exists():
            self.emb = np.load(p)
            if backend == "lsa":
                vec, svd = pickle.load(open(index_dir / "lsa.pkl", "rb"))
                self.enc = LSAEncoder()
                self.enc.vec, self.enc.svd = vec, svd
            else:
                self.enc = make_encoder(backend, EMBED_MODEL, VOYAGE_MODEL)
        self.reranker = None
        if RERANK_MODEL:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL)
        self.backend = backend if self.emb is not None else "none"

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 8, acts: list[str] | None = None, pool: int = 40) -> dict:
        trace: dict = {"query": query}
        det_acts = detect_acts(query)
        act_filter = acts or None
        trace["acts_detected"] = det_acts
        secs, scheds = explicit_citations(query)
        trace["explicit"] = {"sections": secs, "schedules": scheds}
        q_exp = expand(query)
        trace["expanded"] = q_exp if q_exp != query else None
        allowed = [i for i, c in enumerate(self.chunks) if not act_filter or c["act"] in act_filter]
        allowed_set = set(allowed)

        bm = {i: s for i, s in self.bm25.scores(tokenize(q_exp)).items() if i in allowed_set}
        bt = {i: s for i, s in self.bm25_title.scores(tokenize(q_exp)).items() if i in allowed_set}
        bm_rank = sorted(bm, key=bm.get, reverse=True)[:pool]
        bt_rank = sorted(bt, key=bt.get, reverse=True)[:pool]
        dn_rank, dn = [], {}
        if self.emb is not None:
            qv = self.enc.encode_query(q_exp)[0]
            sims = self.emb @ qv
            dn = {i: float(sims[i]) for i in allowed}
            dn_rank = sorted(dn, key=dn.get, reverse=True)[:pool]

        W = {"bm25": 1.0, "title": 0.5, "dense": 1.0 if dn_rank else 0.0}
        rrf: dict[int, float] = {}
        for name, rank in (("bm25", bm_rank), ("title", bt_rank), ("dense", dn_rank)):
            for r, i in enumerate(rank):
                rrf[i] = rrf.get(i, 0.0) + W[name] / (60 + r + 1)
        # boost chunks from an Act named in the query
        if det_acts and not act_filter:
            for i in rrf:
                if self.chunks[i]["act"] in det_acts:
                    rrf[i] *= 1.25
        hints = [a for a in act_hints(query) if a not in det_acts]
        trace["act_hints"] = hints
        sched_intent = bool(SCHEDULE_INTENT.search(query))
        def_intent = bool(DEFINITION_INTENT.search(query))
        for i in rrf:
            c = self.chunks[i]
            if c["act"] in hints:
                rrf[i] *= 1.2
            if sched_intent and c["kind"] in ("schedule", "schedule_table"):
                rrf[i] *= 1.15
            if def_intent and c["kind"] == "definition":
                rrf[i] *= 1.15
        # pin explicit citations
        pinned: list[int] = []
        for n in secs:
            for a in (det_acts or [c for c in ("NTA", "NTAA", "NRSEA", "JRBEA")]):
                for c in self.by_sec.get(f"{a}:s{n}", []):
                    idx = self._idx(c["id"])
                    if idx in allowed_set:
                        pinned.append(idx)
                if det_acts:
                    continue
        for o in scheds:
            for a in (det_acts or ["NTA", "NTAA", "NRSEA", "JRBEA"]):
                for c in self.by_sec.get(f"{a}:sched:{o}", [])[:3]:
                    idx = self._idx(c["id"])
                    if idx in allowed_set:
                        pinned.append(idx)
        # undetected act + explicit section number: pin only the Act(s) whose section title overlaps query terms
        if secs and not det_acts and pinned:
            qt = set(tokenize(q_exp))
            keep = []
            for idx in pinned:
                c = self.chunks[idx]
                if len(qt & set(tokenize(c["title"]))) > 0 or len({self.chunks[j]["act"] for j in pinned}) == 1:
                    keep.append(idx)
            pinned = keep or pinned[:2]
        for idx in pinned:
            rrf[idx] = rrf.get(idx, 0) + 1.0
        ranked = sorted(rrf, key=rrf.get, reverse=True)[: max(k * 3, 20)]

        if self.reranker and ranked:
            pairs = [(query, self.chunks[i]["embed_text"][:2000]) for i in ranked]
            sc = self.reranker.predict(pairs)
            order = sorted(range(len(ranked)), key=lambda j: sc[j], reverse=True)
            ranked = [ranked[j] for j in order]
            trace["reranked"] = True
            rr_scores = {ranked[j]: float(sc[order[j]]) for j in range(len(ranked))}
        else:
            rr_scores = {}
        # keep pinned in front
        front = [i for i in dict.fromkeys(pinned) if i in ranked][:3]
        ranked = front + [i for i in ranked if i not in front]
        top = ranked[:k]
        results = []
        for i in top:
            c = self.chunks[i]
            results.append({**c, "scores": {
                "rrf": round(rrf.get(i, 0), 5), "bm25": round(bm.get(i, 0), 3), "dense": round(dn.get(i, 0), 3),
                "rerank": round(rr_scores.get(i, 0), 3) if rr_scores else None, "pinned": i in pinned}})
        trace["backend"] = self.backend
        return {"results": results, "trace": trace}

    def _idx(self, cid: str) -> int:
        if not hasattr(self, "_idmap"):
            self._idmap = {c["id"]: i for i, c in enumerate(self.chunks)}
        return self._idmap[cid]

    # ----------------------------------------------------------- context build
    def assemble(self, query: str, results: list[dict], max_chars: int = 14000, max_refs: int = 4, max_defs: int = 4) -> dict:
        """Turn ranked chunks into the numbered source list given to the LLM."""
        sources: list[dict] = []
        seen: set[str] = set()
        used = 0

        def add(c: dict, role: str):
            nonlocal used
            if c["id"] in seen or used + len(c["text"]) > max_chars:
                return False
            seen.add(c["id"])
            sources.append({"n": len(sources) + 1, "role": role, **{k: c[k] for k in (
                "id", "act", "act_title", "kind", "label", "sub", "title", "chapter", "part", "text", "gaz", "section", "schedule", "term")}})
            used += len(c["text"])
            return True

        primary = [r for r in results]
        for r in primary:
            add(r, "retrieved")
        # stitch: if a section was split, include adjacent pieces of the *top* section so no subsection is cut off
        for r in primary[:3]:
            for nb in (r.get("prev"), r.get("next")):
                if nb and nb in self.by_id and r["kind"] == "section":
                    add(self.by_id[nb], "adjacent")
        # follow cross-references from the top results
        n_ref = 0
        for r in primary[:4]:
            for ref in r.get("refs", []):
                for c in self.by_sec.get(ref, [])[:1]:
                    if n_ref < max_refs and add(c, "cross-reference"):
                        n_ref += 1
        # attach definitions of specific defined terms that appear in the question or retrieved text
        blob = (query + " " + " ".join(r["text"] for r in primary[:3])).lower()
        qlow = query.lower()
        cand = []
        for term, entries in self.defs.items():
            if term in GENERIC_TERMS or len(term) < 5:
                continue
            if re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", blob):
                cand.append((term in qlow, len(term), term, entries))
        cand.sort(reverse=True)
        n_def = 0
        for _, _, term, entries in cand:
            if n_def >= max_defs:
                break
            for e in entries:
                if self.by_id[e["id"]]["act"] and add(self.by_id[e["id"]], "definition"):
                    n_def += 1
                    break
        return {"sources": sources}
