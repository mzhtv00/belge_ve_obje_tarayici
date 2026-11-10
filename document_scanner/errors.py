"""Exceptions raised by the document scanning package."""


class ScannerError(Exception):
    """Base exception for expected scanner failures."""


class InvalidImageError(ScannerError, ValueError):
    """An input image is empty or has an unsupported shape or data type."""


class DocumentNotFoundError(ScannerError):
    """No sufficiently large four-corner document boundary was found."""


class ImageProcessingError(ScannerError):
    """An image could not be transformed or exported."""


class OCRBackendError(ScannerError):
    """An OCR backend is unavailable or failed to recognize text."""


class DatabaseError(ScannerError):
    """The SQLite store could not initialize or complete an operation."""
