"""Lazy, pluggable OCR backends for Turkish and English."""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import cv2
import numpy as np

from ._image import validate_image
from .errors import OCRBackendError

Language = Literal["tur", "eng", "tur+eng", "eng+tur"]


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float | None = None


@runtime_checkable
class OCRBackend(Protocol):
    def recognize(self, image: np.ndarray) -> OCRResult:
        """Recognize text in a grayscale or OpenCV BGR image."""


def _to_gray(image: np.ndarray) -> np.ndarray:
    validate_image(image)
    if image.ndim == 2 or image.shape[2] == 1:
        return image.reshape(image.shape[0], image.shape[1])
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class TesseractBackend:
    """Tesseract adapter; requires pytesseract and the Tesseract executable."""

    def __init__(self, language: Language = "tur+eng", *, config: str = "") -> None:
        if language not in ("tur", "eng", "tur+eng", "eng+tur"):
            raise ValueError("language must be tur, eng, tur+eng, or eng+tur.")
        self.language = language
        self.config = config

    def recognize(self, image: np.ndarray) -> OCRResult:
        gray = _to_gray(image)
        try:
            import pytesseract

            text = pytesseract.image_to_string(
                gray, lang=self.language, config=self.config
            )
        except ImportError as error:
            raise OCRBackendError(
                "Tesseract support is not installed. Install with "
                "`pip install document-scanner[tesseract]`."
            ) from error
        except Exception as error:
            raise OCRBackendError(
                "Tesseract OCR failed. Check the Tesseract executable and language "
                f"data: {error}"
            ) from error
        return OCRResult(text=text.strip())


class EasyOCRBackend:
    """EasyOCR adapter; imports its model and torch only when recognition starts."""

    def __init__(
        self, languages: tuple[str, ...] = ("tr", "en"), *, gpu: bool = False
    ) -> None:
        if not languages or any(language not in ("tr", "en") for language in languages):
            raise ValueError("languages must contain 'tr', 'en', or both.")
        self.languages = tuple(dict.fromkeys(languages))
        self.gpu = gpu
        self._reader = None

    def recognize(self, image: np.ndarray) -> OCRResult:
        validate_image(image)
        if image.ndim == 2 or image.shape[2] == 1:
            ocr_image = image.reshape(image.shape[0], image.shape[1])
        elif image.shape[2] == 4:
            ocr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            ocr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            import easyocr

            if self._reader is None:
                self._reader = easyocr.Reader(list(self.languages), gpu=self.gpu)
            results = self._reader.readtext(ocr_image, detail=1, paragraph=False)
        except ImportError as error:
            raise OCRBackendError(
                "EasyOCR support is not installed. Install with "
                "`pip install document-scanner[easyocr]`."
            ) from error
        except Exception as error:
            raise OCRBackendError(
                f"EasyOCR failed to recognize the image: {error}"
            ) from error

        lines: list[str] = []
        confidences: list[float] = []
        for result in results:
            if len(result) >= 3 and str(result[1]).strip():
                lines.append(str(result[1]).strip())
                confidences.append(float(result[2]))
        confidence = sum(confidences) / len(confidences) if confidences else None
        return OCRResult(text="\n".join(lines), confidence=confidence)


def create_ocr_backend(
    name: Literal["tesseract", "easyocr"] = "tesseract",
    *,
    language: Language = "tur+eng",
    gpu: bool = False,
) -> OCRBackend:
    """Construct an OCR backend without importing its optional package."""
    if name == "tesseract":
        return TesseractBackend(language=language)
    if name == "easyocr":
        if language not in ("tur", "eng", "tur+eng", "eng+tur"):
            raise ValueError("language must be tur, eng, tur+eng, or eng+tur.")
        languages = tuple("tr" if code == "tur" else "en" for code in language.split("+"))
        return EasyOCRBackend(languages, gpu=gpu)
    raise ValueError("name must be 'tesseract' or 'easyocr'.")
