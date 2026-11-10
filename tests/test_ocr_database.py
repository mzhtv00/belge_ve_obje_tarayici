import sys
from types import SimpleNamespace

import numpy as np
import pytest

from document_scanner import (
    DatabaseError,
    EasyOCRBackend,
    InvalidImageError,
    OCRBackendError,
    SQLiteDocumentStore,
    TesseractBackend,
    create_ocr_backend,
)


class FakeReader:
    def readtext(self, image, *, detail, paragraph):
        assert detail == 1
        assert paragraph is False
        return [([], "Merhaba dünya", 0.9), ([], "Invoice 42", 0.8)]


def test_fts_store_search_update_and_literal_query(tmp_path) -> None:
    with SQLiteDocumentStore(tmp_path / "documents.sqlite3") as store:
        document_id = store.upsert_document(
            "scan.png", "Merhaba dünya — invoice 42", document_date="2026-11-11"
        )

        results = store.search('merhaba "dünya"')

        assert len(results) == 1
        assert results[0].id == document_id
        assert results[0].file_path == "scan.png"
        assert results[0].document_date == "2026-11-11"
        assert store.upsert_document("scan.png", "Güncellenmiş fatura") == document_id
        assert store.search("merhaba") == []
        assert store.search("güncellenmiş fatura")[0].text == "Güncellenmiş fatura"


def test_search_validation_and_closed_store(tmp_path) -> None:
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    with pytest.raises(ValueError):
        store.search("test", limit=0)
    store.close()
    with pytest.raises(DatabaseError):
        store.search("test")


def test_ocr_backend_construction_is_lazy_and_maps_languages() -> None:
    tesseract = create_ocr_backend("tesseract", language="tur")
    easyocr = create_ocr_backend("easyocr", language="tur+eng")

    assert isinstance(tesseract, TesseractBackend)
    assert isinstance(easyocr, EasyOCRBackend)
    assert easyocr.languages == ("tr", "en")


def test_easyocr_results_and_ocr_input_validation(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=lambda *a, **k: FakeReader()))
    backend = EasyOCRBackend()
    result = backend.recognize(np.zeros((20, 30, 3), dtype=np.uint8))

    assert result.text == "Merhaba dünya\nInvoice 42"
    assert result.confidence == pytest.approx(0.85)
    with pytest.raises(InvalidImageError):
        backend.recognize(np.array([], dtype=np.uint8))


def test_unavailable_optional_ocr_dependency_is_reported() -> None:
    with pytest.raises(OCRBackendError, match="Tesseract support is not installed"):
        TesseractBackend().recognize(np.zeros((10, 10), dtype=np.uint8))
