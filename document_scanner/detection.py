"""Document boundary detection using Canny edges and contour analysis."""

import cv2
import numpy as np

from ._image import validate_image
from .errors import DocumentNotFoundError


def detect_document(
    image: np.ndarray,
    *,
    min_area_ratio: float = 0.05,
    max_dimension: int = 1200,
) -> np.ndarray:
    """Return the largest detected quadrilateral in source-image coordinates."""
    validate_image(image)
    if not 0 < min_area_ratio <= 1:
        raise ValueError("min_area_ratio must be greater than 0 and at most 1.")
    if max_dimension < 32:
        raise ValueError("max_dimension must be at least 32 pixels.")

    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / float(max(height, width)))
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    if resized.ndim == 3 and resized.shape[2] == 4:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
    elif resized.ndim == 3 and resized.shape[2] == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized.reshape(resized.shape[0], resized.shape[1])

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    min_area = resized.shape[0] * resized.shape[1] * min_area_ratio
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            break
        polygon = cv2.approxPolyDP(
            contour, 0.02 * cv2.arcLength(contour, True), True
        )
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            return (polygon.reshape(4, 2).astype(np.float32) / scale).astype(
                np.float32
            )

    raise DocumentNotFoundError(
        "No four-corner document boundary was found; provide a clearer image "
        "or lower min_area_ratio."
    )
