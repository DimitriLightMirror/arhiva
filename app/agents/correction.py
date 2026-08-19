"""Agent 1 - OCR correction agent.

Receives raw OCR text per block with low-confidence words flagged and asks
an LLM to produce corrected Romanian text: restore diacritics (a-acute,
s/t-comma), fix classic OCR confusions (rn->m, 0->O, l->I) and normalize
legal terminology. With LLM_PROVIDER=none (or on any LLM error) the raw
text is returned unchanged so the pipeline never breaks.
"""

from __future__ import annotations

import re

from ..config import Settings
from ..llm.client import LLMClient, NullClient
from ..ocr_engine import OcrPageResult

SYSTEM_PROMPT = (
    "Esti un expert in corectarea textelor OCR din documente notariale romanesti. "
    "Primesti text OCR brut in care cuvintele cu incredere scazuta sunt marcate "
    "cu [[?cuvant]]. Corecteaza diacriticele romanesti (a cu breve, a cu circumflex, "
    "i cu circumflex, s si t cu virgula dedesubt), confuziile tipice OCR "
    "(rn->m, 0->O, l->I, cedila in loc de virgula) si terminologia juridica. "
    "Nu adauga si nu sterge informatii. Pastreaza structura pe randuri. "
    "Raspunde DOAR cu textul corectat, fara explicatii, fara marcaje [[?...]]."
)

_FLAG_RE = re.compile(r"\[\[\?(.+?)\]\]")


def flag_low_confidence_text(page: OcrPageResult, threshold: float) -> str:
    """Rebuild block text, wrapping low-confidence words in [[?word]] markers."""
    lines: dict[tuple[int, int, int], list[str]] = {}
    for w in page.words:
        token = f"[[?{w.text}]]" if w.conf < threshold else w.text
        lines.setdefault((w.block, w.par, w.line), []).append(token)
    return "\n".join(" ".join(parts) for parts in lines.values())


def strip_flags(text: str) -> str:
    """Remove [[?...]] markers, keeping the inner word."""
    return _FLAG_RE.sub(r"\1", text)


class CorrectionAgent:
    """LLM-backed corrector for Romanian notary OCR text."""

    def __init__(self, llm: LLMClient, settings: Settings):
        self.llm = llm
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return not isinstance(self.llm, NullClient)

    def correct_block(self, raw_text: str) -> str:
        """Correct one text block; on any failure return the raw text."""
        raw_text = raw_text.strip()
        if not raw_text or not self.enabled:
            return strip_flags(raw_text)
        try:
            corrected = self.llm.complete(SYSTEM_PROMPT, raw_text)
            corrected = corrected.strip()
            # Guard against empty / degenerate model output.
            if not corrected or len(corrected) < len(strip_flags(raw_text)) * 0.3:
                return strip_flags(raw_text)
            return corrected
        except Exception:
            return strip_flags(raw_text)

    def correct_page(self, page: OcrPageResult) -> str:
        """Correct a full page, chunking long pages into blocks."""
        flagged = flag_low_confidence_text(page, self.settings.ocr_low_conf_threshold)
        if not flagged.strip():
            return page.text

        max_chars = self.settings.llm_max_block_chars
        if len(flagged) <= max_chars:
            return self.correct_block(flagged)

        # Chunk by lines so we never cut a line in half.
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for line in flagged.splitlines():
            if current_len + len(line) > max_chars and current:
                chunks.append("\n".join(current))
                current, current_len = [], 0
            current.append(line)
            current_len += len(line) + 1
        if current:
            chunks.append("\n".join(current))
        return "\n".join(self.correct_block(c) for c in chunks)
