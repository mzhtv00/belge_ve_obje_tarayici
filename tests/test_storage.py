import io

import pytest
from PIL import Image

from app.storage import InvalidImage, InvalidModel, validate_image, validate_stl


def test_image_validation_rejects_empty_and_unsupported_payload():
    with pytest.raises(InvalidImage):
        validate_image(b"", 1024, 1000)
    with pytest.raises(InvalidImage):
        validate_image(b"not an image", 1024, 1000)


def test_image_validation_enforces_byte_and_pixel_limits():
    image = Image.new("RGB", (24, 16), color=(30, 180, 130))
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    data = stream.getvalue()

    with pytest.raises(InvalidImage, match="exceeds"):
        validate_image(data, len(data) - 1, 1000)
    with pytest.raises(InvalidImage, match="dimensions"):
        validate_image(data, len(data), 100)


def test_stl_validation_accepts_binary_and_rejects_malformed_data():
    triangle = b"\x00" * 50
    valid_binary = b"binary STL header".ljust(80, b"\x00") + (1).to_bytes(4, "little") + triangle
    validate_stl(valid_binary, 1024)
    with pytest.raises(InvalidModel):
        validate_stl(b"solid broken", 1024)
    with pytest.raises(InvalidModel):
        validate_stl(valid_binary, 100)


def test_stl_validation_rejects_ascii_without_facet_structure():
    malformed_ascii = b"solid model\nthis is not a facet\nendsolid model\n"
    valid_ascii = b"""solid model
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
endsolid model
"""
    validate_stl(valid_ascii, 1024)
    with pytest.raises(InvalidModel):
        validate_stl(malformed_ascii, 1024)


def test_binary_stl_rejects_empty_or_non_finite_triangles():
    empty_stl = b"binary STL header".ljust(80, b"\x00") + (0).to_bytes(4, "little")
    invalid_triangle = b"binary STL header".ljust(80, b"\x00") + (1).to_bytes(4, "little")
    invalid_triangle += bytes.fromhex("0000c07f") + b"\x00" * 46

    with pytest.raises(InvalidModel):
        validate_stl(empty_stl, 1024)
    with pytest.raises(InvalidModel):
        validate_stl(invalid_triangle, 1024)
