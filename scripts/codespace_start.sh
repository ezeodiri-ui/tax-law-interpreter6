#!/usr/bin/env bash
# Starts the app inside a GitHub Codespace (or any Linux shell). Safe to run repeatedly.
set -e
cd "$(dirname "$0")/../backend"
export DENSE_BACKEND="${DENSE_BACKEND:-st}"
export RERANK_MODEL="${RERANK_MODEL-BAAI/bge-reranker-v2-m3}"
export LLM_PROVIDER="${LLM_PROVIDER:-groq}"
export LLM_MODEL="${LLM_MODEL:-openai/gpt-oss-120b}"
[ -f ../.data_ready_"$DENSE_BACKEND" ] || { python -m app.ingest.build_index && touch ../.data_ready_"$DENSE_BACKEND"; }
pkill -f "uvicorn app.main:app" 2>/dev/null || true
setsid nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > ../server.log 2>&1 &
sleep 6
echo "Server started. Log: server.log"
curl -s localhost:8000/healthz >/dev/null && echo "Healthy. Open the 'Ports' tab and open port 8000."
[ -z "$GROQ_API_KEY" ] && echo "NOTE: GROQ_API_KEY is not set, so the app will run in retrieval-only mode."
[ -z "$APP_PASSWORD" ] && echo "NOTE: APP_PASSWORD is not set, so anyone with the link can use the app."
true
