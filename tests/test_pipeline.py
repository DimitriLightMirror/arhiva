"""Pipeline wiring test with OCR and LLM fully mocked.

Verifies that all stages run in order, that region/correction/classification
outputs flow into the final payload, and that artifacts are routed into the
archive tree /archive/<class>/<year>/<doc_id>/.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from app.agents.classification import ClassificationResult
from app.config import Settings
from app.ocr_engine import OcrPageResult, WordBox
from app.pipeline import DocumentPipeline, JobRegistry


class FakeLLM:
    """LLM double: correction echoes a marker, classification returns JSON."""

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        if json_mode:
            return json.dumps(
                {
                    "document_class": "procura",
                    "confidence": 0.88,
                    "fields": {"parti": ["MARIA POP"], "numar_cadastral": "",
                               "data": "01.02.2024", "adresa": "", "notar": "",
                               "valoare": ""},
                    "tags": ["<parte>", "<data>"],
                    "tagged_text": "<parte>MARIA POP</parte>",
                }
            )
        return "TEXT CORECTAT: " + user.replace("[[?", "").replace("]]", "")


def _fake_ocr_page(page_number: int) -> OcrPageResult:
    words = [
        WordBox("PROCURA", 10, 10, 100, 20, 95.0, 1, 1, 1, 1),
        WordBox("autentica", 10, 40, 120, 20, 45.0, 1, 1, 2, 1),
    ]
    return OcrPageResult(
        page_number=page_number, width=800, height=1100,
        text="PROCURA\nautentica", words=words, mean_confidence=70.0,
        lang_used="ron",
    )


@pytest.fixture()
def pipeline(tmp_path, monkeypatch):
    settings = Settings(
        llm_provider="none",
        archive_root=str(tmp_path / "archive"),
        watch_folder=str(tmp_path / "watch"),
        data_dir=str(tmp_path / "data"),
    )
    pipe = DocumentPipeline(settings)
    pipe.classification.llm = FakeLLM()
    pipe.correction.llm = FakeLLM()

    # Mock rasterization and OCR (no tesseract / PDF needed).
    page_image = np.full((1100, 800, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(pipe, "rasterize", lambda b, f: [page_image])
    monkeypatch.setattr(pipe, "ocr_pages", lambda pre: [_fake_ocr_page(1)])
    return pipe, settings


def test_pipeline_runs_all_stages(pipeline):
    pipe, settings = pipeline
    registry = JobRegistry()
    job = registry.create("procura_scan.png")

    pipe.run(job, b"fake-bytes", registry)

    assert job.status == "done", job.error
    assert job.stage == "done"
    result = job.result
    assert result["num_pages"] == 1

    # Correction agent ran through the LLM double.
    page = result["pages"][0]
    assert page["corrected_text"].startswith("TEXT CORECTAT:")
    assert page["raw_text"] == "PROCURA\nautentica"

    # Layout produced regions for the page.
    assert isinstance(page["regions"], list)
    assert all("type" in r for r in page["regions"])

    # Classification flowed into the payload.
    assert result["classification"]["document_class"] == "procura"
    assert result["classification"]["fields"]["parti"] == ["MARIA POP"]


def test_pipeline_archives_artifacts(pipeline):
    pipe, settings = pipeline
    registry = JobRegistry()
    job = registry.create("procura_scan.png")
    pipe.run(job, b"fake-bytes", registry)

    pdf_path = Path(job.pdf_path)
    assert pdf_path.exists() and pdf_path.name == "searchable.pdf"
    assert pdf_path.read_bytes().startswith(b"%PDF")

    # Archive routing: <archive>/procura/<year>/<doc_id>/
    year_dirs = list((Path(settings.archive_root) / "procura").iterdir())
    assert len(year_dirs) == 1
    assert pdf_path.parent.parent == year_dirs[0]

    sidecar = pdf_path.with_name("result.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["job_id"] == job.id
    assert payload["classification"]["document_class"] == "procura"


def test_pipeline_reports_errors(pipeline, monkeypatch):
    pipe, _ = pipeline
    monkeypatch.setattr(
        pipe, "rasterize", lambda b, f: (_ for _ in ()).throw(ValueError("bad file"))
    )
    registry = JobRegistry()
    job = registry.create("broken.pdf")
    pipe.run(job, b"junk", registry)
    assert job.status == "error"
    assert "bad file" in job.error
