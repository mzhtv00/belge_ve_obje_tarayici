import pytest

import photogrammetry_pipeline as pipeline


def test_config_normalizes_paths_and_rejects_invalid_limits(tmp_path) -> None:
    config = pipeline.PipelineConfig(tmp_path / "photos", tmp_path / "output")

    assert config.image_dir == tmp_path / "photos"
    assert config.output_dir == tmp_path / "output"
    with pytest.raises(ValueError):
        pipeline.PipelineConfig(tmp_path, tmp_path, min_images=1)


def test_image_discovery_is_recursive_and_filters_extensions(tmp_path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "photo.JPG").touch()
    (tmp_path / "scan.png").touch()
    (tmp_path / "notes.txt").touch()

    images = pipeline.ColmapReconstructor._find_images(tmp_path)

    assert [path.name for path in images] == ["photo.JPG", "scan.png"]


def test_optional_processing_dependency_failure_is_clear(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "o3d", None)

    with pytest.raises(pipeline.PipelineError, match="open3d"):
        pipeline.PointCloudProcessor(
            pipeline.PipelineConfig("photos", "output")
        )
