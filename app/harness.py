import time
from typing import Optional, Dict

from app.generator import AnswerGenerator
from app.guardrails import grounding_check, input_guard, retrieval_guard
from app.latency import timed_stage
from app.retriever import HybridRetriever
from app.schemas import RagResponse
from app.stt_sarvam import SarvamSTT


class VoiceRAGHarness:
    def __init__(self):
        self.stt = SarvamSTT()
        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator()

    def ask_audio(self, audio_bytes: bytes, filename: str, content_type: str) -> RagResponse:
        timings: Dict[str, float] = {}
        total_start = time.perf_counter()

        with timed_stage("stt_ms", timings):
            try:
                transcript = self.stt.transcribe(audio_bytes, filename, content_type)
            except Exception as e:
                timings["total_ms"] = round((time.perf_counter() - total_start) * 1000, 3)
                return RagResponse(
                    transcript="",
                    answer="Speech-to-text transcription failed.",
                    citations=[],
                    grounded=False,
                    abstained=True,
                    abstain_reason=f"STT Error: {str(e)}",
                    timings_ms=timings,
                )

        response = self.ask_text(transcript, timings)

        response.timings_ms["total_ms"] = round(
            (time.perf_counter() - total_start) * 1000,
            3,
        )

        return response

    def ask_text(self, query: str, timings: Optional[Dict[str, float]] = None) -> RagResponse:
        timings = timings or {}
        total_start = time.perf_counter()
        rag_start = time.perf_counter()

        transcript = (query or "").strip()

        with timed_stage("input_guard_ms", timings):
            ok, reason = input_guard(transcript)

        if not ok:
            timings["post_stt_total_ms"] = round((time.perf_counter() - rag_start) * 1000, 3)
            if "total_ms" not in timings:
                timings["total_ms"] = timings["post_stt_total_ms"]
            return RagResponse(
                transcript=transcript,
                answer="I cannot process that request.",
                citations=[],
                grounded=True,
                abstained=True,
                abstain_reason=reason,
                timings_ms=timings,
            )

        with timed_stage("retrieval_ms", timings):
            contexts, confidence = self.retriever.retrieve(transcript)

        with timed_stage("retrieval_guard_ms", timings):
            ok, reason = retrieval_guard(contexts, confidence)

        if not ok:
            timings["post_stt_total_ms"] = round((time.perf_counter() - rag_start) * 1000, 3)
            if "total_ms" not in timings:
                timings["total_ms"] = timings["post_stt_total_ms"]
            return RagResponse(
                transcript=transcript,
                answer="I could not find enough relevant context in the MSMARCO-XI dataset to answer this.",
                citations=[],
                grounded=True,
                abstained=True,
                abstain_reason=reason,
                timings_ms=timings,
            )

        with timed_stage("generation_ms", timings):
            answer, citations = self.generator.generate_extractive(transcript, contexts)

        with timed_stage("grounding_ms", timings):
            grounded = grounding_check(answer, contexts)

        if not grounded:
            answer = "I found related passages, but I cannot produce a sufficiently grounded answer from them."
            citations = []
            abstained = True
            reason = "Grounding validation failed."
        else:
            abstained = False
            reason = None

        timings["post_stt_total_ms"] = round((time.perf_counter() - rag_start) * 1000, 3)
        if "total_ms" not in timings:
            timings["total_ms"] = timings["post_stt_total_ms"]

        return RagResponse(
            transcript=transcript,
            answer=answer,
            citations=citations,
            grounded=grounded,
            abstained=abstained,
            abstain_reason=reason,
            timings_ms=timings,
        )
