"""Quadrilateral ordering and perspective correction."""

import cv2
import numpy as np

from ._image import validate_image
from .errors import ImageProcessingError


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four 2D points as top-left, top-right, bottom-right, bottom-left."""
    coordinates = np.asarray(points, dtype=np.float32)
    if coordinates.shape != (4, 2) or not np.isfinite(coordinates).all():
        raise ValueError("points must be a finite array with shape (4, 2).")

    sums = coordinates.sum(axis=1)
    differences = np.diff(coordinates, axis=1).reshape(4)
    ordered = np.array(
        [
            coordinates[np.argmin(sums)],
            coordinates[np.argmin(differences)],
            coordinates[np.argmax(sums)],
            coordinates[np.argmax(differences)],
        ],
        dtype=np.float32,
    )
    if len(np.unique(ordered, axis=0)) != 4:
        raise ValueError("points must contain four distinct corners.")
    return ordered


def rectify_document(
    image: np.ndarray,
    points: np.ndarray,
    *,
    max_output_dimension: int = 12000,
) -> np.ndarray:
    """Warp a detected quadrilateral into a top-down rectangular image."""
    validate_image(image)
    if max_output_dimension < 1:
        raise ValueError("max_output_dimension must be positive.")
    top_left, top_right, bottom_right, bottom_left = order_points(points)

    width = int(
        round(
            max(
                np.linalg.norm(bottom_right - bottom_left),
                np.linalg.norm(top_right - top_left),
            )
        )
    )
    height = int(
        round(
            max(
                np.linalg.norm(top_right - bottom_right),
                np.linalg.norm(top_left - bottom_left),
            )
        )
    )
    if width < 2 or height < 2:
        raise ImageProcessingError("Document boundaries are too small to rectify.")
    if max(width, height) > max_output_dimension:
        raise ImageProcessingError(
            f"Rectified dimensions {width}x{height} exceed the "
            f"{max_output_dimension}-pixel limit."
        )

    source = np.array(
        [top_left, top_right, bottom_right, bottom_left], dtype=np.float32
    )
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(image, transform, (width, height))
    if warped.size == 0:
        raise ImageProcessingError("Perspective transformation produced an empty image.")
    return warped
