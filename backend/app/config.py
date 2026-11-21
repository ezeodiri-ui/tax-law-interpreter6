import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = Path(os.getenv("PDF_DIR", ROOT / "data" / "pdfs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))
FRONTEND_DIR = ROOT / "frontend"

# Dense backend: "st" (sentence-transformers, local), "voyage" (API, voyage-law-2), "lsa" (offline fallback), "none"
DENSE_BACKEND = os.getenv("DENSE_BACKEND", "lsa")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-law-2")
# Cross-encoder reranker ("" disables). Recommended: BAAI/bge-reranker-v2-m3
RERANK_MODEL = os.getenv("RERANK_MODEL", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
TOP_K = int(os.getenv("TOP_K", "8"))
