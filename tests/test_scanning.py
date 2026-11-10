import cv2
import numpy as np
import pytest

from document_scanner import (
    DocumentProcessingPipeline,
    DocumentNotFoundError,
    ImageProcessingError,
    InvalidImageError,
    OCRResult,
    SQLiteDocumentStore,
    detect_document,
    enhance_document,
    export_document,
    order_points,
    rectify_document,
)


def make_page_image() -> np.ndarray:
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    page = np.array([[60, 35], [340, 55], [320, 260], [40, 240]], dtype=np.int32)
    cv2.fillConvexPoly(image, page, (235, 235, 235))
    cv2.line(image, (95, 100), (285, 110), (30, 30, 30), 4)
    return image


def test_detect_and_rectify_document() -> None:
    image = make_page_image()
    corners = detect_document(image)
    rectified = rectify_document(image, corners)

    assert corners.shape == (4, 2)
    assert rectified.shape[0] > 180
    assert rectified.shape[1] > 250


def test_order_points_returns_consistent_corner_order() -> None:
    points = np.array([[10, 10], [80, 15], [75, 90], [5, 85]], dtype=np.float32)

    assert np.array_equal(order_points(points[[2, 0, 3, 1]]), points)


def test_detection_rejects_invalid_or_missing_input() -> None:
    with pytest.raises(InvalidImageError):
        detect_document(np.array([], dtype=np.uint8))
    with pytest.raises(DocumentNotFoundError):
        detect_document(np.zeros((100, 100, 3), dtype=np.uint8))


def test_enhancement_modes_preserve_dimensions() -> None:
    image = make_page_image()

    color = enhance_document(image, mode="color")
    grayscale = enhance_document(image, mode="grayscale")
    binary = enhance_document(image, mode="binary")

    assert color.shape == image.shape
    assert grayscale.shape == image.shape[:2]
    assert binary.shape == image.shape[:2]
    assert set(np.unique(binary)).issubset({0, 255})


def test_export_png_and_pdf(tmp_path) -> None:
    image = np.full((32, 48, 3), (10, 80, 180), dtype=np.uint8)

    png_path = export_document(image, tmp_path / "nested" / "scan.png")
    pdf_path = export_document(image, tmp_path / "scan.pdf")

    decoded = cv2.imread(str(png_path))
    assert decoded is not None and decoded.shape == image.shape
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_export_and_warp_reject_unsupported_limits() -> None:
    image = make_page_image()
    with pytest.raises(ValueError):
        export_document(image, "scan.jpg")
    with pytest.raises(ImageProcessingError):
        rectify_document(
            image,
            np.array([[0, 0], [399, 0], [399, 299], [0, 299]], dtype=np.float32),
            max_output_dimension=100,
        )


def test_pipeline_exports_ocr_text_and_indexes_document(tmp_path) -> None:
    class FakeOCR:
        def recognize(self, image):
            return OCRResult("Test document text", 0.95)

    image = make_page_image()
    with SQLiteDocumentStore(tmp_path / "documents.sqlite3") as store:
        result = DocumentProcessingPipeline(FakeOCR(), store).process(
            image,
            source_path="camera.jpg",
            output_path=tmp_path / "scan.png",
            document_date="2026-11-11",
        )

        assert result.output_path.is_file()
        assert result.text == "Test document text"
        assert result.confidence == 0.95
        assert store.search("document text")[0].file_path == "camera.jpg"
