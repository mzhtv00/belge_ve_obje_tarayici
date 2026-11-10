import cv2
import numpy as np
import pytest

import object_scan_pipeline as pipeline


def make_config(work_dir, **kwargs):
    return pipeline.PipelineConfig(
        work_dir=work_dir,
        min_images=2,
        max_images=4,
        **kwargs,
    )


def write_image(path):
    assert cv2.imwrite(str(path), np.full((16, 20, 3), 127, dtype=np.uint8))


def test_config_rejects_invalid_processing_limits(tmp_path) -> None:
    config = make_config(tmp_path / "work")

    assert config.work_dir == tmp_path / "work"
    with pytest.raises(pipeline.InputValidationError):
        make_config(tmp_path, voxel_size=0)
    with pytest.raises(pipeline.InputValidationError):
        make_config(tmp_path, poisson_depth=15)
    with pytest.raises(pipeline.InputValidationError):
        make_config(tmp_path, reconstruction_method="unknown")


def test_image_set_validates_and_stages_supported_files(tmp_path) -> None:
    image_dir = tmp_path / "photos"
    image_dir.mkdir()
    write_image(image_dir / "B.JPG")
    write_image(image_dir / "a.png")
    (image_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    image_set = pipeline.ImageSet.from_directory(image_dir, make_config(tmp_path / "work"))
    destination = image_set.stage(tmp_path / "staged")

    assert [path.name for path in image_set.paths] == ["a.png", "B.JPG"]
    assert sorted(path.name for path in destination.iterdir()) == ["B.JPG", "a.png"]


def test_image_set_rejects_corrupt_images(tmp_path) -> None:
    image_dir = tmp_path / "photos"
    image_dir.mkdir()
    write_image(image_dir / "good.png")
    (image_dir / "corrupt.jpg").write_bytes(b"not an image")

    with pytest.raises(pipeline.InputValidationError, match="corrupt or unsupported"):
        pipeline.ImageSet.from_directory(image_dir, make_config(tmp_path / "work"))


def test_image_set_enforces_count_and_file_size_limits(tmp_path) -> None:
    image_dir = tmp_path / "photos"
    image_dir.mkdir()
    write_image(image_dir / "one.jpg")
    with pytest.raises(pipeline.InputValidationError, match="Expected"):
        pipeline.ImageSet.from_directory(image_dir, make_config(tmp_path / "work"))

    write_image(image_dir / "two.jpg")
    config = make_config(tmp_path / "work", max_image_bytes=1)
    with pytest.raises(pipeline.InputValidationError, match="exceeds size limit"):
        pipeline.ImageSet.from_directory(image_dir, config)


def test_missing_open3d_has_actionable_error(monkeypatch, tmp_path) -> None:
    def missing_package(_package_name):
        raise pipeline.DependencyError("Missing Python dependency 'open3d'.")

    monkeypatch.setattr(pipeline, "_require_package", missing_package)
    processor = pipeline.PointCloudProcessor(make_config(tmp_path / "work"))

    with pytest.raises(pipeline.DependencyError, match="open3d"):
        processor.process(tmp_path / "cloud.ply", tmp_path / "cleaned.ply")
