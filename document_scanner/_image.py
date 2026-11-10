"""Shared validation for OpenCV image arrays."""

import numpy as np

from .errors import InvalidImageError


def validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise InvalidImageError("Image must be a NumPy array.")
    if image.size == 0 or image.ndim not in (2, 3):
        raise InvalidImageError("Image must be a non-empty grayscale or color array.")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise InvalidImageError("Color images must have 1, 3, or 4 channels.")
    if image.dtype != np.uint8:
        raise InvalidImageError("Image data must use the uint8 data type.")
    return image
