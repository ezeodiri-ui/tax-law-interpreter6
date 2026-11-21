# Tax Laws Interpreter (RAG) – House Committee on Finance demo

Question answering over four 2025 Acts: Nigeria Tax Act (NTA), Nigeria Tax Administration Act (NTAA),
Nigeria Revenue Service (Establishment) Act (NRSEA), Joint Revenue Board of Nigeria (Establishment) Act (JRBEA).
Every answer cites numbered provisions (Act, section/schedule, Gazette page) and is machine-checked against them.

## Run
    cp .env.example .env      # add ANTHROPIC_API_KEY; choose DENSE_BACKEND
    ./run.sh                  # builds the index, serves http://localhost:8000
    # or: docker compose up --build
With no model configured the app runs in retrieval-only mode (ranked provisions, no generated answer).

## Pipeline
1. **Extraction** (`app/ingest/pdf_extract.py`): removes Gazette marginal notes and running headers; keeps the printed
   Gazette page; restores the naira sign (the PDFs drop it, leaving a gap before amounts); rebuilds tables, including
   schedules printed sideways; repairs hyphenation.
2. **Structure** (`structure.py`): Arrangement of Sections is the master list; sections are located in order, giving
   Chapter > Part > Section > subsection > paragraph, plus every Schedule. 455 of 455 sections and 23 schedules found.
3. **Chunking** (`chunker.py`): short sections whole; long ones split on subsection boundaries with lead-in text carried;
   one chunk per definition; schedule tables one row per record with column labels; each chunk embeds a context header
   (Act > Chapter > Part > Section title); prev/next links; cross-references extracted as graph edges.
4. **Retrieval** (`app/retrieval/engine.py`): BM25 over context-headed text + BM25 over titles + dense vectors, fused by
   RRF; explicit citations ("section 148", "Ninth Schedule") pinned; Act/institution hints; legacy-term expansion
   (FIRS, CITA, WHT...); optional cross-encoder rerank.
5. **Context assembly**: sibling chunks stitched, cross-referenced provisions followed, definitions of specific defined
   terms attached.
6. **Generation** (`app/generation/llm.py`): model-agnostic. Choose the model with `LLM_PROVIDER` / `LLM_MODEL` /
   key variables (see `.env.example`); Groq, Gemini, OpenRouter, Ollama, any OpenAI-compatible server (vLLM, LM Studio)
   and Anthropic are supported, and switching is a settings change only. Temperature 0; answers only from the numbered
   sources in a fixed format. Free-model names change often: if a default fails, set `LLM_MODEL` from the provider's list.
7. **Verification** (`verify.py`): flags invalid citation numbers, quotations not found verbatim in the sources, and
   figures (₦ amounts, percentages) absent from the sources.

## Evaluate
    cd backend && python -m eval.run_eval --backend lsa     # or none | st | voyage
`eval/gold.jsonl` is a 32-question hand-built set (question -> acceptable provisions). Extend it with the Committee's
real questions before relying on the numbers.

## Measured here (offline; no model downloads)
| backend | Recall@1 | @3 | @5 | @8 | MRR |
|---|---|---|---|---|---|
| keyword only (BM25) | 0.59 | 0.81 | 0.88 | 0.97 | 0.72 |
| BM25 + LSA fallback | 0.56 | 0.81 | 0.88 | 0.97 | 0.69 |

Not run in the build environment (no model downloads / API key): `st` and `voyage` dense backends, the cross-encoder
reranker, and generation with real providers (the streaming adapter and verification were tested only against a local mock server). Run the evaluation with `DENSE_BACKEND=st RERANK_MODEL=BAAI/bge-reranker-v2-m3` on
your machine and use those numbers for the Committee.

## Known limitations
- Hint rules and the gold set were tuned on the same small question set; expect lower scores on unseen questions.
- The Tenth Schedule has no extracted title. Gazette typos (e.g. "Thirtenth Schedule", "Nineth") are preserved in text.
- The system answers from the text only; it does not know of amendments, regulations or court decisions after these Acts.
- Table rows were checked for coverage, not read line by line; proofread schedules before quoting them publicly.
