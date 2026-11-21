import base64
import hmac
import json
import os
import time
from collections import defaultdict, deque
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .config import FRONTEND_DIR, TOP_K, DENSE_BACKEND
from .retrieval.engine import Engine
from .generation import llm
from .generation.verify import verify

app = FastAPI(title="Tax Laws RAG \u2013 Committee on Finance demo")
engine = Engine()

# ---- access control & fair use (set APP_PASSWORD to switch on) ---------------------------
APP_USER = os.getenv("APP_USER", "committee")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))  # /api/ask + /api/search per visitor
_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def guard(request: Request, call_next):
    if APP_PASSWORD and request.url.path != "/healthz":
        ok = False
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("basic "):
            try:
                user, _, pw = base64.b64decode(auth[6:]).decode().partition(":")
                ok = hmac.compare_digest(user, APP_USER) and hmac.compare_digest(pw, APP_PASSWORD)
            except Exception:
                ok = False
        if not ok:
            return Response("Authentication required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="Committee demo"'})
    if request.url.path in ("/api/ask", "/api/search"):
        ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
        q, now = _hits[ip], time.time()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return Response("Too many requests; please wait a minute.", status_code=429)
        q.append(now)
    return await call_next(request)


@app.get("/healthz")
def healthz():
    return {"ok": True}


class AskReq(BaseModel):
    query: str
    acts: list[str] | None = None
    k: int = TOP_K
    history: list[dict] | None = None


def _public(c: dict) -> dict:
    return {k: v for k, v in c.items() if k not in ("embed_text",)}


@app.get("/api/health")
def health():
    return {"ok": True, "chunks": len(engine.chunks), "dense_backend": engine.backend, "reranker": bool(engine.reranker),
            "llm": llm.available(), "llm_name": llm.describe()}


@app.post("/api/search")
def search(req: AskReq):
    r = engine.search(req.query, k=req.k, acts=req.acts)
    return {"trace": r["trace"], "results": [_public(x) for x in r["results"]]}


@app.post("/api/ask")
def ask(req: AskReq):
    r = engine.search(req.query, k=req.k, acts=req.acts)
    ctx = engine.assemble(req.query, r["results"])
    sources = ctx["sources"]

    def gen():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'trace': r['trace']})}\n\n"
        if not llm.available():
            yield f"data: {json.dumps({'type': 'notice', 'text': 'No answer model configured (set LLM_PROVIDER and its key): showing retrieved provisions only.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'verification': None})}\n\n"
            return
        buf = []
        try:
            for t in llm.stream_answer(req.query, sources, req.history):
                buf.append(t)
                yield f"data: {json.dumps({'type': 'token', 'text': t})}\n\n"
        except Exception as e:  # surface API errors to the UI
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'verification': verify(''.join(buf), sources)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/models")
def models():
    """Models the configured provider offers to your key (use to pick LLM_MODEL)."""
    try:
        return {"current": llm.describe(), "available": llm.list_models()}
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/toc")
def toc():
    return engine.toc


@app.get("/api/provision/{act}/{kind}/{ref}")
def provision(act: str, kind: str, ref: str):
    key = f"{act}:s{ref}" if kind == "section" else f"{act}:sched:{ref}"
    cs = engine.by_sec.get(key)
    if not cs:
        raise HTTPException(404, "not found")
    return {"chunks": [_public(c) for c in cs]}


app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
