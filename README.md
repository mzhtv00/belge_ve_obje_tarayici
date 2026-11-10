# Document scanning and photogrammetry

Python 3.11+ modules for scanning photographed documents into corrected,
searchable files and reconstructing 3D models from turntable photo sets.

## Install

Install the base dependencies (document processing) with:

```powershell
python -m pip install .
```

OCR engines are optional and loaded only when selected:

```powershell
python -m pip install ".[tesseract]"  # Also install the Tesseract executable and tur/eng language data.
python -m pip install ".[easyocr]"    # EasyOCR may download its models on first recognition.
```

For 3D reconstruction, install COLMAP separately and install the Python
dependencies with `python -m pip install ".[photogrammetry]"`. The combined
`requirements.txt` also lists the 3D dependencies.

## 2D document scanning

Images are NumPy `uint8` arrays in grayscale or OpenCV BGR/BGRA channel order.
The pipeline locates a four-corner page with Canny edges and contour analysis,
rectifies its perspective, corrects uneven lighting, exports PNG or PDF, runs
OCR, then saves the source path, optional document date, and extracted text to
SQLite FTS5.

```python
import cv2

from document_scanner import (
    DocumentProcessingPipeline,
    SQLiteDocumentStore,
    create_ocr_backend,
)

image = cv2.imread("photo.jpg", cv2.IMREAD_COLOR)
if image is None:
    raise ValueError("Could not read photo.jpg")

with SQLiteDocumentStore("data/documents.sqlite3") as store:
    pipeline = DocumentProcessingPipeline(
        create_ocr_backend("tesseract", language="tur+eng"),
        store,
        enhancement_mode="color",
    )
    result = pipeline.process(
        image,
        source_path="photo.jpg",
        output_path="scans/photo.png",  # Use a .pdf extension for PDF output.
        document_date="2026-11-11",     # Optional ISO-8601 date.
    )
    print(result.output_path, result.document_id, result.text)

    for document in store.search("fatura ödeme"):
        print(document.file_path, document.document_date, document.text)
```

Choose `create_ocr_backend("easyocr")` for EasyOCR using Turkish and English.
`language="eng"` or `"tur"` selects one Tesseract language. EasyOCR accepts
`languages=("en",)` or `("tr",)` when constructed directly. Search treats input
as literal words, requiring every query term to match; FTS operators are not
passed through as executable search syntax. Updating an existing source path
updates the matching FTS index row.

Image operations are available individually:

```python
from document_scanner import detect_document, enhance_document, rectify_document

corners = detect_document(image)
top_down = rectify_document(image, corners)
cleaned = enhance_document(top_down, mode="binary")
```

Enhancement modes are `color`, `grayscale`, and `binary`. Invalid images,
missing boundaries, image export failures, unavailable/failed OCR engines, and
SQLite/FTS5 failures raise explicit package exceptions. Invalid configuration
raises `ValueError`.

## 3D photogrammetry

`photogrammetry_pipeline.py` reconstructs a dense point cloud from overlapping
object photos with COLMAP, then filters the cloud and creates, repairs,
decimates, and exports a mesh with Open3D and Trimesh.

Install the Python dependencies with `python -m pip install ".[photogrammetry]"`
or `python -m pip install -r requirements.txt`. COLMAP is a separate
application and must be available on `PATH` or passed as an executable path.

Place 15–36 supported photos (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, or
`.bmp`) in an image directory. Capture a complete orbit with substantial
overlap, diffuse lighting, and a textured/matte object and background.

```powershell
python .\photogrammetry_pipeline.py .\photos .\output
```

The output directory receives `model.stl`, `model.obj`, and a `work` directory
containing COLMAP's intermediate files. Use `--method ball_pivoting` to select
Ball Pivoting instead of Poisson, `--colmap` to provide COLMAP's full path, or
`--help` to see additional processing options.

The default mesh cap is 250,000 triangles, and both exported files are kept
under 25 MB (25,000,000 bytes). `--max-faces` and `--max-file-size-mb` tune
these limits. Hole repair fills simple boundary loops but cannot guarantee a
watertight result for every scan; inspect the exported mesh before printing.

## Tests

```powershell
python -m pip install ".[test]"
python -m pytest
```
