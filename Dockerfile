# Works on Hugging Face Spaces (Docker SDK), Render, Railway, Fly.io, or any VPS.
FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app
COPY --chown=user backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r backend/requirements.txt sentence-transformers
COPY --chown=user . .
WORKDIR /home/user/app/backend
ENV DENSE_BACKEND=st RERANK_MODEL=BAAI/bge-reranker-v2-m3 LLM_PROVIDER=groq LLM_MODEL=openai/gpt-oss-120b PORT=7860
# Build the index and pre-download both models at build time so start-up is fast.
RUN python -m app.ingest.build_index \
 && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
EXPOSE 7860
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
