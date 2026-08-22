FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# sentence-transformers pulls in torch transitively, and PyPI's default torch
# wheel bundles NVIDIA CUDA libraries even though this app only ever runs
# CPU inference (Cloud Run has no GPU). Installing the CPU-only wheel first
# means the later requirements.txt install finds torch already satisfied and
# never pulls the ~7GB of unused CUDA packages, shrinking build time and
# image size substantially.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image at build time. Downloading it from
# Hugging Face on every container *start* (instead of once at build) added
# ~100s to cold start, which pushed total startup past Cloud Run's 4-minute
# startup probe timeout.
ENV HF_HOME=/app/.hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

COPY . .
RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000

CMD ["scripts/entrypoint.sh"]
