"""Lighting correction and scan-like image enhancement."""

from typing import Literal

import cv2
import numpy as np

from ._image import validate_image

EnhancementMode = Literal["color", "grayscale", "binary"]


def enhance_document(
    image: np.ndarray,
    *,
    mode: EnhancementMode = "color",
    clahe_clip_limit: float = 2.0,
    adaptive_block_size: int = 31,
    adaptive_c: int = 15,
) -> np.ndarray:
    """Correct uneven lighting, apply CLAHE, and optionally adaptive-threshold.

    ``color`` enhances luminance while preserving the original color. ``grayscale``
    returns a corrected grayscale image; ``binary`` additionally applies adaptive
    thresholding. Every mode preserves the input dimensions.
    """
    validate_image(image)
    if mode not in ("color", "grayscale", "binary"):
        raise ValueError("mode must be 'color', 'grayscale', or 'binary'.")
    if clahe_clip_limit <= 0:
        raise ValueError("clahe_clip_limit must be positive.")
    if adaptive_block_size < 3 or adaptive_block_size % 2 == 0:
        raise ValueError("adaptive_block_size must be an odd number of at least 3.")

    if image.ndim == 2 or image.shape[2] == 1:
        gray, color = image.reshape(image.shape[0], image.shape[1]).copy(), None
    elif image.shape[2] == 4:
        color = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    else:
        color = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    illumination = cv2.GaussianBlur(gray, (0, 0), sigmaX=21, sigmaY=21)
    corrected = cv2.divide(gray, np.maximum(illumination, 1), scale=255)
    corrected = cv2.createCLAHE(
        clipLimit=clahe_clip_limit, tileGridSize=(8, 8)
    ).apply(corrected)

    if mode == "binary":
        return cv2.adaptiveThreshold(
            corrected,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            adaptive_block_size,
            adaptive_c,
        )
    if mode == "grayscale" or color is None:
        return corrected

    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(
        clipLimit=clahe_clip_limit, tileGridSize=(8, 8)
    ).apply(lightness)
    return cv2.cvtColor(
        cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )
