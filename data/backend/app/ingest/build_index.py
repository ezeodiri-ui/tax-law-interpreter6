"""python -m app.ingest.build_index  -> parses PDFs, chunks, embeds, writes data/index/"""
import json
import sys
import numpy as np
from ..config import PDF_DIR, INDEX_DIR, DENSE_BACKEND, EMBED_MODEL, VOYAGE_MODEL
from .structure import ACTS, parse_act
from .chunker import build_chunks, extract_definitions
from ..retrieval.dense import make_encoder


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    parsed = {code: parse_act(code, str(PDF_DIR / meta["file"])) for code, meta in ACTS.items()}
    chunks = build_chunks(parsed)
    toc = []
    for code, (secs, scheds) in parsed.items():
        for s in secs:
            toc.append({"act": code, "type": "section", "number": s.number, "title": s.title, "chapter": s.chapter, "part": s.part})
        for sc in scheds:
            toc.append({"act": code, "type": "schedule", "number": sc.ordinal, "title": sc.title, "chapter": "", "part": ""})
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    (INDEX_DIR / "toc.json").write_text(json.dumps(toc, ensure_ascii=False))
    (INDEX_DIR / "definitions.json").write_text(json.dumps(extract_definitions(chunks), ensure_ascii=False))
    enc = make_encoder(DENSE_BACKEND, EMBED_MODEL, VOYAGE_MODEL)
    meta = {"backend": DENSE_BACKEND, "n_chunks": len(chunks)}
    if enc is not None:
        emb = enc.encode_docs([c["embed_text"] for c in chunks])
        np.save(INDEX_DIR / "embeddings.npy", emb)
        if DENSE_BACKEND == "lsa":
            import pickle
            pickle.dump((enc.vec, enc.svd), open(INDEX_DIR / "lsa.pkl", "wb"))
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta))
    print(f"Indexed {len(chunks)} chunks with dense backend '{DENSE_BACKEND}' -> {INDEX_DIR}")


if __name__ == "__main__":
    main()
