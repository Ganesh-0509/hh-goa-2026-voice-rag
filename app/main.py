import os
from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.harness import VoiceRAGHarness
from app.schemas import RagResponse, TextRequest

app = FastAPI(
    title="HH Goa 2026 - Voice RAG MSMARCO-XI",
    description="Low-latency voice-enabled grounded RAG system with Sarvam STT, Qdrant, SQLite FTS5, and multi-strategy chunking.",
    version="1.0.0",
)

# Public demo endpoint - a per-IP limit deters basic scripted abuse (each
# request costs real compute, and /api/ask-audio calls the paid Sarvam API)
# without getting in the way of a judge trying the demo interactively.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

harness: VoiceRAGHarness | None = None


@app.on_event("startup")
def startup_event():
    global harness
    print("Initializing VoiceRAGHarness and preloading embedding model...")
    harness = VoiceRAGHarness()
    print("VoiceRAGHarness initialized successfully.")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "harness_loaded": harness is not None,
    }


@app.post("/api/ask-text", response_model=RagResponse)
@limiter.limit("20/minute")
def ask_text(request: Request, req: TextRequest):
    if not harness:
        raise HTTPException(status_code=503, detail="Harness not initialized")
    return harness.ask_text(req.query)


@app.post("/api/ask-audio", response_model=RagResponse)
@limiter.limit("20/minute")
async def ask_audio(request: Request, file: UploadFile = File(...)):
    if not harness:
        raise HTTPException(status_code=503, detail="Harness not initialized")

    audio_bytes = await file.read()
    filename = file.filename or "audio.webm"
    content_type = file.content_type or "audio/webm"

    return harness.ask_audio(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
    )


# Serve web interface
if os.path.exists("web"):
    app.mount("/static", StaticFiles(directory="web"), name="static")


@app.get("/")
def serve_ui():
    if os.path.exists("web/index.html"):
        return FileResponse("web/index.html")
    return {"message": "Voice RAG API is running. Web UI file index.html not found."}
