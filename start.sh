#!/bin/sh
# Start the API in the background and the UI in the foreground (the UI calls the API).
set -e
uvicorn src.api:app --host 0.0.0.0 --port 8000 &
exec streamlit run app/Ask.py --server.address=0.0.0.0 --server.port="${PORT:-7860}"
