#!/usr/bin/env bash
# Local run:  ./run.sh        (reads .env if present)
set -e
cd "$(dirname "$0")/backend"
[ -f ../.env ] && set -a && . ../.env && set +a
pip install -q -r requirements.txt
python -m app.ingest.build_index
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
