"""python -m eval.run_eval [--backend lsa|none|st|voyage]   (run from backend/)
Hand-built gold set: question -> acceptable provision id prefixes. Reports Recall@k and MRR."""
import argparse, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.retrieval.engine import Engine


def run(backend: str, ks=(1, 3, 5, 8), verbose=False):
    e = Engine(backend=backend)
    gold = [json.loads(l) for l in open(pathlib.Path(__file__).with_name("gold.jsonl")) if l.strip()]
    hits = {k: 0 for k in ks}
    rr = 0.0
    misses = []
    for g in gold:
        res = e.search(g["q"], k=max(ks))["results"]
        rank = next((i + 1 for i, r in enumerate(res) if any(r["id"].startswith(p) and (len(r["id"]) == len(p) or r["id"][len(p)] in "-") for p in g["expect"])), None)
        if rank:
            rr += 1 / rank
            for k in ks:
                hits[k] += rank <= k
        else:
            misses.append(g["q"])
        if verbose or not rank or rank > 3:
            print(f"  rank={rank}  {g['q']}")
    n = len(gold)
    out = {"backend": e.backend, "n": n, **{f"recall@{k}": round(hits[k] / n, 3) for k in ks}, "mrr": round(rr / n, 3)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="lsa")
    a = ap.parse_args()
    print(json.dumps(run(a.backend), indent=2))
