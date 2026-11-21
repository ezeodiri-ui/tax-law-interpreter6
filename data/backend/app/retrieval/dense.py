"""Pluggable dense encoders. All return L2-normalised float32 matrices."""
from __future__ import annotations
import os
import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _norm(m: np.ndarray) -> np.ndarray:
    m = m.astype("float32")
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


class STEncoder:
    name = "st"

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model
        self.m = SentenceTransformer(model)

    def encode_docs(self, texts):
        return _norm(self.m.encode(texts, batch_size=32, show_progress_bar=True))

    def encode_query(self, q):
        pre = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return _norm(self.m.encode([pre + q]))


class VoyageEncoder:
    name = "voyage"

    def __init__(self, model: str):
        import voyageai
        self.c = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        self.model = model

    def encode_docs(self, texts):
        out = []
        for i in range(0, len(texts), 64):
            out.extend(self.c.embed(texts[i:i + 64], model=self.model, input_type="document").embeddings)
        return _norm(np.array(out))

    def encode_query(self, q):
        return _norm(np.array(self.c.embed([q], model=self.model, input_type="query").embeddings))


class LSAEncoder:
    """Offline fallback: TF-IDF + truncated SVD. Needs no model download."""
    name = "lsa"

    def __init__(self, dim=256):
        self.dim = dim

    def fit(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from .text import tokenize
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True, ngram_range=(1, 2), min_df=2)
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(min(self.dim, X.shape[1] - 1), random_state=0)
        return _norm(self.svd.fit_transform(X))

    def encode_docs(self, texts):
        return self.fit(texts)

    def encode_query(self, q):
        return _norm(self.svd.transform(self.vec.transform([q])))


def make_encoder(backend: str, model: str, voyage_model: str):
    if backend == "st":
        return STEncoder(model)
    if backend == "voyage":
        return VoyageEncoder(voyage_model)
    if backend == "lsa":
        return LSAEncoder()
    return None
