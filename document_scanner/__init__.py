"""Document scanning, enhancement, OCR, and searchable storage."""

from .database import DocumentRecord, SQLiteDocumentStore
from .detection import detect_document
from .enhancement import enhance_document
from .errors import (
    DatabaseError,
    DocumentNotFoundError,
    ImageProcessingError,
    InvalidImageError,
    OCRBackendError,
    ScannerError,
)
from .export import export_document
from .ocr import (
    EasyOCRBackend,
    OCRBackend,
    OCRResult,
    TesseractBackend,
    create_ocr_backend,
)
from .perspective import order_points, rectify_document
from .pipeline import DocumentProcessingPipeline, ProcessingResult

__all__ = [
    "DatabaseError",
    "DocumentNotFoundError",
    "DocumentProcessingPipeline",
    "DocumentRecord",
    "EasyOCRBackend",
    "ImageProcessingError",
    "InvalidImageError",
    "OCRBackend",
    "OCRBackendError",
    "OCRResult",
    "ProcessingResult",
    "SQLiteDocumentStore",
    "ScannerError",
    "TesseractBackend",
    "create_ocr_backend",
    "detect_document",
    "enhance_document",
    "export_document",
    "order_points",
    "rectify_document",
]
