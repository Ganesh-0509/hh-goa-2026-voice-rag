import re
import regex
from typing import List, Tuple

from app.schemas import Citation, RetrievedContext

# NOTE: stdlib re's Unicode \w does NOT include combining marks (Mn/Mc categories),
# so it shreds Indic-script conjuncts (Devanagari matras/virama, and equivalents in
# Bengali/Gujarati/Tamil/etc.) into single-character fragments - e.g. "कॉर्पोरेशन"
# (corporation) becomes ['क','र','प','र','शन',...] instead of one token. That broke
# term-overlap scoring for the entire (non-English) MSMARCO-XI corpus. The `regex`
# package's \p{L}\p{M}\p{N} properly keeps letter+mark+digit runs together.
_WORD_PATTERN = regex.compile(r"[\p{L}\p{M}\p{N}]+", flags=regex.UNICODE)


def split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?।॥])\s+", text)


def important_terms(query: str) -> set[str]:
    terms = _WORD_PATTERN.findall(query.lower())
    # Exclude common short stop-ish words
    return {t for t in terms if len(t) > 2 and t not in {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "that", "this", "these", "those"}}


class AnswerGenerator:
    def generate_extractive(
        self,
        query: str,
        contexts: List[RetrievedContext],
    ) -> Tuple[str, List[Citation]]:
        q_terms = important_terms(query)

        candidates = []

        for ctx_rank, ctx in enumerate(contexts):
            for sent in split_sentences(ctx.text):
                sent = sent.strip()
                if len(sent) < 25:
                    continue

                s_terms = set(_WORD_PATTERN.findall(sent.lower()))
                overlap = len(q_terms & s_terms)

                # Prioritize sentence term overlap, then context rank
                score = (overlap * 3.0) + (1.0 / (ctx_rank + 1))

                candidates.append((score, sent, ctx))

        if not candidates:
            return "", []

        candidates.sort(key=lambda x: x[0], reverse=True)

        selected = []
        used = set()

        for score, sent, ctx in candidates:
            norm = sent.lower()
            if norm in used:
                continue

            selected.append((sent, ctx))
            used.add(norm)

            if len(selected) >= 2:
                break

        if not selected:
            return "", []

        answer_sentences = [s for s, _ in selected]
        answer = " ".join(answer_sentences)

        citations = []
        for sent, ctx in selected:
            citations.append(
                Citation(
                    chunk_id=ctx.chunk_id,
                    score=ctx.score,
                    strategy=ctx.strategy,
                    language=ctx.language,
                    quote=sent[:400],
                )
            )

        return answer, citations
