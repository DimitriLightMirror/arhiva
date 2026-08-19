"""Agent 2 - document classification, field extraction and tagging.

Classifies the full document into Romanian notary document classes, extracts
key fields (parti, numar cadastral, data, adresa, notar, valoare) and emits
XML-style balises/tags around important entities in the text.

The LLM is instructed to answer with strict JSON; the response is parsed
defensively (code fences, leading/trailing prose) so a slightly misbehaving
model cannot crash the pipeline. With LLM_PROVIDER=none the classifier
falls back to a keyword heuristic so routing still works.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..llm.client import LLMClient, NullClient

# Canonical document classes; the folder names are used for archive routing.
DOCUMENT_CLASSES: dict[str, str] = {
    "act_vanzare_cumparare": "Act de vanzare-cumparare",
    "contract_ipoteca": "Contract de ipoteca",
    "certificat_urbanism": "Certificat de urbanism",
    "extras_carte_funciara": "Extras de carte funciara",
    "certificat_performanta_energetica": "Certificat de performanta energetica",
    "plan_cadastral": "Plan cadastral / plan de parcela",
    "procura": "Procura",
    "incheiere": "Incheiere",
    "alt_document": "Alt document",
}

KEY_FIELDS = ["parti", "numar_cadastral", "data", "adresa", "notar", "valoare"]

SYSTEM_PROMPT = (
    "Esti un expert in documente notariale romanesti. Clasifici documente scanate "
    "si extragi campuri cheie. Clasele posibile (foloseste exact cheia):\n"
    + "\n".join(f"- {key}: {label}" for key, label in DOCUMENT_CLASSES.items())
    + "\n\nRaspunde STRICT cu un obiect JSON valid, fara text suplimentar, "
    "cu structura:\n"
    "{\n"
    '  "document_class": "<cheie clasa>",\n'
    '  "confidence": <numar 0..1>,\n'
    '  "fields": {"parti": [...], "numar_cadastral": "", "data": "", '
    '"adresa": "", "notar": "", "valoare": ""},\n'
    '  "tags": ["<balize XML-style, ex: <parte>, <numar_cadastral>, '
    '<data>, <adresa>, <notar>, <valoare>"],\n'
    '  "tagged_text": "<fragment reprezentativ din document cu entitatile '
    'inconajurate de balize XML-style, ex: <parte>ION POPESCU</parte>>"\n'
    "}\n"
    "Campurile lipsa se completeaza cu sir gol sau lista goala."
)


@dataclass
class ClassificationResult:
    document_class: str = "alt_document"
    class_label: str = "Alt document"
    confidence: float = 0.0
    fields: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    tagged_text: str = ""
    agent_used: str = "none"  # "llm" | "heuristic" | "none"

    def to_dict(self) -> dict:
        return {
            "document_class": self.document_class,
            "class_label": self.class_label,
            "confidence": round(self.confidence, 3),
            "fields": self.fields,
            "tags": self.tags,
            "tagged_text": self.tagged_text,
            "agent_used": self.agent_used,
        }


def parse_classification_json(raw: str) -> ClassificationResult:
    """Parse the LLM JSON response defensively into a ClassificationResult."""
    text = raw.strip()
    # Strip Markdown code fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Keep only the outermost JSON object if there is surrounding prose.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in classification response")
    data = json.loads(text[start : end + 1])

    doc_class = str(data.get("document_class", "alt_document")).strip()
    if doc_class not in DOCUMENT_CLASSES:
        doc_class = "alt_document"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    raw_fields = data.get("fields") or {}
    fields = {k: raw_fields.get(k, [] if k == "parti" else "") for k in KEY_FIELDS}

    tags = data.get("tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    tags = [str(t) for t in tags]

    return ClassificationResult(
        document_class=doc_class,
        class_label=DOCUMENT_CLASSES[doc_class],
        confidence=confidence,
        fields=fields,
        tags=tags,
        tagged_text=str(data.get("tagged_text", "")),
        agent_used="llm",
    )


# --- Heuristic fallback (LLM_PROVIDER=none) -------------------------------

_HEURISTIC_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("extras_carte_funciara", ("carte funciara", "extras de carte funciara", "ancpi")),
    ("certificat_urbanism", ("certificat de urbanism",)),
    ("certificat_performanta_energetica", ("performanta energetica", "certificat energetic", "audit energetic")),
    ("contract_ipoteca", ("ipoteca", "contract de ipoteca", "creditor ipotecar")),
    ("procura", ("procura", "imputernicire", "mandatar")),
    ("act_vanzare_cumparare", ("vanzare", "cumparare", "pret de vanzare", "vanzator", "cumparator")),
    ("incheiere", ("incheiere", "incuviintare", "executorie")),
    ("plan_cadastral", ("plan cadastral", "plan de parcela", "parcela", "tarla")),
]


def heuristic_classify(text: str, has_plan_region: bool = False) -> ClassificationResult:
    """Keyword-based fallback classifier used when no LLM is configured."""
    low = text.lower()
    best_class, best_hits = "alt_document", 0
    for cls, keywords in _HEURISTIC_RULES:
        hits = sum(low.count(k) for k in keywords)
        if hits > best_hits:
            best_class, best_hits = cls, hits
    if best_class == "alt_document" and has_plan_region and re.search(
        r"parcel|tarla|cadastr", low
    ):
        best_class, best_hits = "plan_cadastral", 1

    confidence = min(0.9, 0.3 + 0.15 * best_hits) if best_hits else 0.1

    # Very light field extraction: cadastral number and dates.
    fields: dict = {k: ([] if k == "parti" else "") for k in KEY_FIELDS}
    m = re.search(r"\b(\d{4,6}[/-][A-Z0-9/-]*)\b", text)
    if m and "cadastr" in low:
        fields["numar_cadastral"] = m.group(1)
    m = re.search(r"\b(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4})\b", text)
    if m:
        fields["data"] = m.group(1)

    return ClassificationResult(
        document_class=best_class,
        class_label=DOCUMENT_CLASSES[best_class],
        confidence=confidence,
        fields=fields,
        tags=[],
        tagged_text="",
        agent_used="heuristic",
    )


class ClassificationAgent:
    """LLM-backed classifier with heuristic fallback."""

    def __init__(self, llm: LLMClient, max_chars: int = 8000):
        self.llm = llm
        self.max_chars = max_chars

    def classify(self, full_text: str, has_plan_region: bool = False) -> ClassificationResult:
        if isinstance(self.llm, NullClient):
            return heuristic_classify(full_text, has_plan_region)
        excerpt = full_text[: self.max_chars]
        try:
            raw = self.llm.complete(SYSTEM_PROMPT, excerpt, json_mode=True)
            return parse_classification_json(raw)
        except Exception:
            # Never let the LLM take the pipeline down.
            return heuristic_classify(full_text, has_plan_region)
