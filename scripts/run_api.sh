#!/bin/bash
set -e

echo "Starting API server..."
export PYTHONPATH=.
uvicorn api.main:app --host 0.0.0.0 --port 8000
