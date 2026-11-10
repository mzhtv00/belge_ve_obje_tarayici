"""PNG and PDF export helpers."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ._image import validate_image
from .errors import ImageProcessingError


def export_document(
    image: np.ndarray,
    output_path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """Write an image as PNG or single-page PDF and return its destination."""
    validate_image(image)
    if dpi < 1:
        raise ValueError("dpi must be a positive integer.")
    destination = Path(output_path)
    suffix = destination.suffix.lower()
    if suffix not in (".png", ".pdf"):
        raise ValueError("output_path must end in .png or .pdf.")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".png":
            if not cv2.imwrite(str(destination), image):
                raise ImageProcessingError(f"OpenCV could not write {destination}.")
        elif image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            Image.fromarray(image.reshape(image.shape[0], image.shape[1])).save(
                destination, format="PDF", resolution=dpi
            )
        elif image.shape[2] == 4:
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)).save(
                destination, format="PDF", resolution=dpi
            )
        else:
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(
                destination, format="PDF", resolution=dpi
            )
    except (OSError, cv2.error) as error:
        raise ImageProcessingError(
            f"Could not export document to {destination}: {error}"
        ) from error
    return destination
