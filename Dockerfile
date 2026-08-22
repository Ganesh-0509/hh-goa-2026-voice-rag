FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
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
