"""End-to-end document image processing and OCR indexing."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .database import SQLiteDocumentStore
from .detection import detect_document
from .enhancement import EnhancementMode, enhance_document
from .export import export_document
from .ocr import OCRBackend, OCRResult
from .perspective import rectify_document


@dataclass(frozen=True)
class ProcessingResult:
    output_path: Path
    document_id: int
    text: str
    confidence: float | None
    image: np.ndarray


class DocumentProcessingPipeline:
    """Rectify, enhance, export, OCR, and index one document image."""

    def __init__(
        self,
        ocr_backend: OCRBackend,
        store: SQLiteDocumentStore,
        *,
        enhancement_mode: EnhancementMode = "color",
    ) -> None:
        self.ocr_backend = ocr_backend
        self.store = store
        self.enhancement_mode = enhancement_mode

    def process(
        self,
        image: np.ndarray,
        *,
        source_path: str | Path,
        output_path: str | Path,
        document_date: date | str | None = None,
    ) -> ProcessingResult:
        corners = detect_document(image)
        rectified = rectify_document(image, corners)
        enhanced = enhance_document(rectified, mode=self.enhancement_mode)
        saved_path = export_document(enhanced, output_path)
        result: OCRResult = self.ocr_backend.recognize(enhanced)
        document_id = self.store.upsert_document(
            source_path, result.text, document_date=document_date
        )
        return ProcessingResult(
            output_path=saved_path,
            document_id=document_id,
            text=result.text,
            confidence=result.confidence,
            image=enhanced,
        )
