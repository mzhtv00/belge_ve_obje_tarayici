import io
import json
import math
import os
import re
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

from PIL import Image, UnidentifiedImageError


IMAGE_FORMATS = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
}
MODEL_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
MAX_THUMBNAIL_SIZE = (480, 360)
STL_FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\Z")


class InvalidImage(ValueError):
    pass


class InvalidModel(ValueError):
    pass


def validate_image(data: bytes, max_bytes: int, max_pixels: int) -> tuple[str, str, int, int]:
    if not data:
        raise InvalidImage("The uploaded image is empty.")
    if len(data) > max_bytes:
        raise InvalidImage(f"Image exceeds the {max_bytes // (1024 * 1024)} MiB limit.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            width, height = image.size
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImage("The upload is not a valid image.") from exc
    if image_format not in IMAGE_FORMATS:
        raise InvalidImage("Only JPEG, PNG, and WebP images are supported.")
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise InvalidImage("Image dimensions are invalid or exceed the pixel limit.")
    mime_type, extension = IMAGE_FORMATS[image_format]
    return mime_type, extension, width, height


def validate_stl(data: bytes, max_bytes: int) -> None:
    if not data:
        raise InvalidModel("The STL file is empty.")
    if len(data) > max_bytes:
        raise InvalidModel(f"STL exceeds the {max_bytes // (1024 * 1024)} MiB limit.")
    if len(data) < 84:
        raise InvalidModel("The file is too small to be a valid STL model.")

    triangle_count = int.from_bytes(data[80:84], "little")
    expected_size = 84 + triangle_count * 50
    if expected_size == len(data):
        if triangle_count == 0:
            raise InvalidModel("The binary STL file does not contain any facets.")
        for offset in range(84, len(data), 50):
            values = struct.unpack_from("<12f", data, offset)
            if not all(math.isfinite(value) for value in values):
                raise InvalidModel("The binary STL file contains non-finite coordinates.")
        return
    try:
        lines = [line.strip().split() for line in data.decode("ascii").splitlines() if line.strip()]
    except UnicodeDecodeError as exc:
        raise InvalidModel("The upload is not a valid binary or ASCII STL file.") from exc
    if len(lines) < 9 or not lines[0] or lines[0][0].casefold() != "solid":
        raise InvalidModel("The upload is not a valid binary or ASCII STL file.")
    if not lines[-1] or lines[-1][0].casefold() != "endsolid":
        raise InvalidModel("The upload is not a valid binary or ASCII STL file.")

    def is_vector(line: list[str], keyword: str, qualifier: str | None = None) -> bool:
        value_index = 1
        if line[0].casefold() != keyword:
            return False
        if qualifier is not None:
            if len(line) < 2 or line[1].casefold() != qualifier:
                return False
            value_index = 2
        if len(line) != value_index + 3:
            return False
        values = line[value_index:]
        if any(not STL_FLOAT_PATTERN.fullmatch(value) for value in values):
            return False
        return all(math.isfinite(float(value)) for value in values)

    index = 1
    triangle_count = 0
    while index < len(lines) - 1:
        if index + 6 >= len(lines) - 1:
            raise InvalidModel("The upload is not a valid binary or ASCII STL file.")
        if (
            not is_vector(lines[index], "facet", "normal")
            or lines[index + 1] != ["outer", "loop"]
            or not all(is_vector(lines[index + offset], "vertex") for offset in (2, 3, 4))
            or [part.casefold() for part in lines[index + 5]] != ["endloop"]
            or [part.casefold() for part in lines[index + 6]] != ["endfacet"]
        ):
            raise InvalidModel("The upload is not a valid binary or ASCII STL file.")
        triangle_count += 1
        index += 7
    if triangle_count == 0:
        raise InvalidModel("The ASCII STL file does not contain any facets.")


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as target:
            target.write(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CaptureStore:
    def __init__(self, data_dir: Path, max_image_pixels: int) -> None:
        self.captures_dir = data_dir / "captures"
        self.thumbnails_dir = data_dir / "thumbnails"
        self.max_image_pixels = max_image_pixels

    def save_capture(self, data: bytes, session_id: str, mode: str, max_bytes: int) -> dict[str, Any]:
        mime_type, extension, width, height = validate_image(data, max_bytes, self.max_image_pixels)
        capture_id = uuid.uuid4().hex
        filename = f"{capture_id}{extension}"
        record: dict[str, Any] = {
            "id": capture_id,
            "filename": filename,
            "mime_type": mime_type,
            "width": width,
            "height": height,
            "session_id": session_id,
            "mode": mode,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "status": "queued",
        }
        _atomic_write(self.captures_dir / filename, data)
        self.write_record(record)
        return record

    def write_record(self, record: dict[str, Any]) -> None:
        _atomic_write(
            self.captures_dir / f"{record['id']}.json",
            json.dumps(record, ensure_ascii=True).encode("utf-8"),
        )

    def update_record(self, capture_id: str, **updates: Any) -> dict[str, Any] | None:
        record = self.get_capture(capture_id)
        if record is None:
            return None
        record.update(updates)
        self.write_record(record)
        return record

    def get_capture(self, capture_id: str) -> dict[str, Any] | None:
        if not MODEL_ID_PATTERN.fullmatch(capture_id):
            return None
        path = self.captures_dir / f"{capture_id}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def thumbnail_path(self, capture_id: str) -> Path | None:
        if self.get_capture(capture_id) is None:
            return None
        path = self.thumbnails_dir / f"{capture_id}.jpg"
        return path if path.is_file() else None

    def list_captures(self) -> list[dict[str, Any]]:
        captures: list[dict[str, Any]] = []
        for path in self.captures_dir.glob("*.json"):
            try:
                captures.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(captures, key=lambda item: item.get("captured_at", ""), reverse=True)

    def create_thumbnail(self, capture_id: str) -> dict[str, Any] | None:
        record = self.get_capture(capture_id)
        if record is None:
            return None
        source = self.captures_dir / record["filename"]
        thumbnail_path = self.thumbnails_dir / f"{capture_id}.jpg"
        with Image.open(source) as image:
            image.thumbnail(MAX_THUMBNAIL_SIZE)
            image.convert("RGB").save(thumbnail_path, format="JPEG", quality=82)
        return self.update_record(capture_id, status="stored", thumbnail=f"{capture_id}.jpg")

    def image_path(self, capture_id: str) -> Path | None:
        record = self.get_capture(capture_id)
        if record is None:
            return None
        return self.captures_dir / record["filename"]


class ModelStore:
    def __init__(self, data_dir: Path) -> None:
        self.models_dir = data_dir / "models"

    def save_model(self, data: bytes, original_name: str, session_id: str) -> dict[str, Any]:
        model_id = uuid.uuid4().hex
        filename = f"{model_id}.stl"
        safe_name = PurePath(original_name.replace("\\", "/")).name[:180] or "model.stl"
        record = {
            "id": model_id,
            "filename": filename,
            "original_name": safe_name,
            "session_id": session_id,
            "size_bytes": len(data),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write(self.models_dir / filename, data)
        try:
            _atomic_write(
                self.models_dir / f"{model_id}.json",
                json.dumps(record, ensure_ascii=True).encode("utf-8"),
            )
        except OSError:
            (self.models_dir / filename).unlink(missing_ok=True)
            raise
        return record

    def get_model_record(self, model_id: str, session_id: str) -> dict[str, Any] | None:
        if not MODEL_ID_PATTERN.fullmatch(model_id):
            return None
        metadata_path = self.models_dir / f"{model_id}.json"
        try:
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            not isinstance(record, dict)
            or record.get("id") != model_id
            or record.get("session_id") != session_id
            or not isinstance(record.get("original_name"), str)
        ):
            return None
        return record if (self.models_dir / f"{model_id}.stl").is_file() else None

    def get_model(self, model_id: str, session_id: str) -> Path | None:
        if self.get_model_record(model_id, session_id) is None:
            return None
        return self.models_dir / f"{model_id}.stl"

    def list_models(self, session_id: str) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for path in self.models_dir.glob("*.json"):
            if not MODEL_ID_PATTERN.fullmatch(path.stem):
                continue
            record = self.get_model_record(path.stem, session_id)
            if record is not None:
                models.append(record)
        return sorted(models, key=lambda item: item["created_at"], reverse=True)
