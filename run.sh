#!/bin/bash
# Start the Disaster XAI-Net dashboard: http://localhost:${PORT:-8000}
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null
python -m uvicorn backend.app:app --port "${PORT:-8000}"
