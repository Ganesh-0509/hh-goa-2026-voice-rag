#!/bin/sh
set -e

INDEX_URL="https://github.com/Ganesh-0509/hh-goa-2026-voice-rag/releases/download/prebuilt-index-v1/prebuilt_index.tar.gz"

if [ ! -f "storage/chunks.sqlite" ]; then
    echo "No local index found — downloading pre-built index from GitHub Releases..."
    mkdir -p storage
    curl -fsSL "$INDEX_URL" -o /tmp/prebuilt_index.tar.gz
    tar -xzf /tmp/prebuilt_index.tar.gz -C storage
    rm -f /tmp/prebuilt_index.tar.gz
    echo "Index ready: $(du -sh storage/chunks.sqlite | cut -f1) chunks.sqlite, $(du -sh storage/qdrant_db | cut -f1) qdrant_db"
else
    echo "Local index already present, skipping download."
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
