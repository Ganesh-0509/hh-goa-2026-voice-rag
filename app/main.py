import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.harness import VoiceRAGHarness
from app.schemas import RagResponse, TextRequest

app = FastAPI(
    title="HH Goa 2026 - Voice RAG MSMARCO-XI",
    description="Low-latency voice-enabled grounded RAG system with Sarvam STT, Qdrant, SQLite FTS5, and multi-strategy chunking.",
    version="1.0.0",
)

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
def ask_text(req: TextRequest):
    if not harness:
        raise HTTPException(status_code=503, detail="Harness not initialized")
    return harness.ask_text(req.query)


@app.post("/api/ask-audio", response_model=RagResponse)
async def ask_audio(file: UploadFile = File(...)):
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
