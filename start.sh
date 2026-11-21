#!/usr/bin/env bash
# Start the app (Codespaces or any Linux/Mac shell). Secrets GROQ_API_KEY and APP_PASSWORD
# must already be set as Codespaces secrets (or exported in the shell).
cd "$(dirname "$0")/backend"
: "${GROQ_API_KEY:?GROQ_API_KEY is not set}"
: "${APP_PASSWORD:?APP_PASSWORD is not set - choose a password before sharing the link}"
export DENSE_BACKEND="${DENSE_BACKEND:-st}" LLM_PROVIDER="${LLM_PROVIDER:-groq}"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
