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
`requirements.txt` includes NumPy, OpenCV headless, Open3D, and Trimesh.

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

`object_scan_pipeline.py` validates a set of object photos, stages them in an
isolated workspace, reconstructs a dense point cloud with COLMAP, then filters
the cloud and creates, repairs, decimates, and exports a mesh with Open3D and
Trimesh.

Install the Python dependencies with `python -m pip install ".[photogrammetry]"`
or `python -m pip install -r requirements.txt`. Install the native COLMAP
application separately and make it available on `PATH`. Dense
`patch_match_stereo` requires a CUDA-enabled COLMAP build and a compatible GPU
and driver. The `--cpu-features` option disables GPU only for SIFT feature
extraction and matching; it does not make dense stereo CPU-only.

Place 15–36 JPEG, PNG, or TIFF photos (`.jpg`, `.jpeg`, `.png`, `.tif`, or
`.tiff`) in an image directory. Capture a complete orbit with substantial
overlap, diffuse lighting, and a textured/matte object and background. Inputs
are validated for file size and image dimensions before being copied to a
per-run workspace.

```powershell
python .\object_scan_pipeline.py --images .\captures --output .\exports --work-dir .\scan-work --method poisson --target-triangles 300000
```

Use `--method ball_pivoting` to select Ball Pivoting instead of Poisson,
`--cpu-features` to disable GPU for SIFT only, `--scale-to-mm` to apply a scale
factor to the resulting model, `--allow-open-mesh` to permit non-watertight
exports, or `--help` for all options. The output directory receives STL and
OBJ files plus a unique work directory with staged photos, COLMAP logs,
intermediate reconstruction files, and the cleaned point cloud.

The default image limits are 25 MiB per photo and 12,000 pixels per edge. The
mesh target is 300,000 triangles and each export is limited to 25,000,000
bytes. These are configurable operational limits, not official Tinkercad
limits or guarantees that a model will import. Watertight output is required
by default; repair fills small holes only, so review failed or critical meshes.
Photogrammetry does not recover absolute physical scale: `--scale-to-mm` applies
a user-provided factor when the source dimensions are known.

## Tests

```powershell
python -m pip install ".[test]"
python -m pytest
```
