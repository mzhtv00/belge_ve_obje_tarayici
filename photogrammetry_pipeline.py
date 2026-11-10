"""Photogrammetry and mesh-processing pipeline for turntable image sets.

The reconstruction stage uses the COLMAP command-line application. Point-cloud
and mesh processing use Open3D and Trimesh.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore[assignment]

try:
    import trimesh
except ImportError:
    trimesh = None  # type: ignore[assignment]

LOGGER = logging.getLogger("photogrammetry_pipeline")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class PipelineError(RuntimeError):
    """Raised when a pipeline stage cannot produce a valid artifact."""


def _require_dependencies(
    *, require_open3d: bool = False, require_trimesh: bool = False
) -> None:
    """Raise a clear error when a processing dependency is not installed."""
    missing = []
    if require_open3d and o3d is None:
        missing.append("open3d")
    if require_trimesh and trimesh is None:
        missing.append("trimesh")
    if missing:
        packages = ", ".join(missing)
        raise PipelineError(
            f"Missing required Python package(s): {packages}. "
            "Install the dependencies with 'python -m pip install "
            ".[photogrammetry]' or 'python -m pip install -r requirements.txt'."
        )


class ExternalToolError(PipelineError):
    """Raised when COLMAP is missing or one of its commands fails."""


@dataclass(slots=True)
class PipelineConfig:
    """Settings for reconstruction, point-cloud processing, and mesh export."""

    image_dir: Path
    output_dir: Path
    colmap_executable: str = "colmap"
    reconstruction_method: Literal["poisson", "ball_pivoting"] = "poisson"
    min_images: int = 15
    max_images: int = 36
    voxel_size: float = 0.0
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    poisson_depth: int = 9
    poisson_trim_fraction: float = 0.02
    max_faces: int = 250_000
    max_file_size_mb: float = 25.0
    min_decimation_faces: int = 100
    command_timeout_seconds: int = 7_200
    use_gpu: bool = False

    def __post_init__(self) -> None:
        """Normalize paths and reject invalid processing settings."""
        self.image_dir = Path(self.image_dir)
        self.output_dir = Path(self.output_dir)
        if self.min_images < 2 or self.max_images < self.min_images:
            raise ValueError("Image-count limits must satisfy 2 <= min_images <= max_images.")
        if self.voxel_size < 0:
            raise ValueError("voxel_size cannot be negative.")
        if self.outlier_neighbors < 2 or self.outlier_std_ratio <= 0:
            raise ValueError("Outlier-removal parameters must be positive.")
        if self.poisson_depth < 4:
            raise ValueError("poisson_depth must be at least 4.")
        if not 0 <= self.poisson_trim_fraction < 1:
            raise ValueError("poisson_trim_fraction must be in [0, 1).")
        if self.max_faces < 4 or self.min_decimation_faces < 4:
            raise ValueError("Face limits must be at least 4.")
        if self.max_file_size_mb <= 0 or self.command_timeout_seconds <= 0:
            raise ValueError("File-size limit and command timeout must be positive.")


class ColmapReconstructor:
    """Run COLMAP feature matching, sparse SfM, and dense stereo reconstruction."""

    def __init__(self, executable: str = "colmap", timeout_seconds: int = 7_200) -> None:
        """Create a runner for the specified COLMAP executable."""
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def reconstruct(self, image_dir: Path, work_dir: Path, use_gpu: bool = False) -> Path:
        """Reconstruct and return the dense point-cloud PLY path."""
        images = self._find_images(image_dir)
        if not images:
            raise PipelineError(f"No supported images found in: {image_dir}")
        if shutil.which(self.executable) is None and not Path(self.executable).is_file():
            raise ExternalToolError(
                f"COLMAP executable '{self.executable}' was not found. Install COLMAP or "
                "set PipelineConfig.colmap_executable to its full path."
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        database_path = work_dir / "database.db"
        sparse_dir = work_dir / "sparse"
        dense_dir = work_dir / "dense"
        sparse_dir.mkdir(exist_ok=True)
        dense_dir.mkdir(exist_ok=True)

        self._run(
            "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
        )
        self._run(
            "exhaustive_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        )
        self._run(
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir),
        )

        sparse_model = self._select_sparse_model(sparse_dir)
        self._run(
            "image_undistorter",
            "--image_path", str(image_dir),
            "--input_path", str(sparse_model),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP",
        )
        self._run(
            "patch_match_stereo",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true",
        )
        cloud_path = dense_dir / "fused.ply"
        self._run(
            "stereo_fusion",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(cloud_path),
        )
        if not cloud_path.is_file() or cloud_path.stat().st_size == 0:
            raise PipelineError(f"COLMAP did not create a non-empty point cloud: {cloud_path}")
        return cloud_path

    @staticmethod
    def _find_images(image_dir: Path) -> list[Path]:
        """Return supported image files in deterministic recursive order."""
        if not image_dir.is_dir():
            raise PipelineError(f"Image directory does not exist: {image_dir}")
        return sorted(
            path for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    @staticmethod
    def _select_sparse_model(sparse_dir: Path) -> Path:
        """Choose the mapper output containing the largest registered model."""
        candidates = [
            path for path in sparse_dir.iterdir()
            if path.is_dir() and (path / "cameras.bin").is_file()
        ]
        if not candidates:
            raise PipelineError(
                "COLMAP mapper produced no sparse model; verify that images overlap "
                "and have sufficient texture."
            )
        return max(candidates, key=lambda path: (path / "images.bin").stat().st_size)

    def _run(self, command: str, *arguments: str) -> None:
        """Run one COLMAP command and include useful diagnostics on failure."""
        invocation = [self.executable, command, *arguments]
        LOGGER.info("Running COLMAP %s", command)
        try:
            result = subprocess.run(
                invocation,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExternalToolError(
                f"COLMAP {command} exceeded the {self.timeout_seconds}s timeout."
            ) from exc
        except OSError as exc:
            raise ExternalToolError(f"Could not start COLMAP {command}: {exc}") from exc
        if result.returncode != 0:
            diagnostic = (result.stderr or result.stdout).strip()
            if len(diagnostic) > 4_000:
                diagnostic = diagnostic[-4_000:]
            raise ExternalToolError(
                f"COLMAP {command} failed with exit code {result.returncode}."
                + (f"\n{diagnostic}" if diagnostic else "")
            )


class PointCloudProcessor:
    """Load, denoise, and reconstruct a surface from a COLMAP point cloud."""

    def __init__(self, config: PipelineConfig) -> None:
        """Keep the shared pipeline settings."""
        _require_dependencies(require_open3d=True)
        self.config = config

    def load_and_clean(self, cloud_path: Path) -> o3d.geometry.PointCloud:
        """Load a PLY, remove statistical outliers, and optionally voxel-downsample."""
        cloud = o3d.io.read_point_cloud(str(cloud_path))
        if cloud.is_empty():
            raise PipelineError(f"Open3D could not load a non-empty point cloud: {cloud_path}")
        if len(cloud.points) <= self.config.outlier_neighbors:
            raise PipelineError(
                f"Point cloud has only {len(cloud.points)} points; "
                f"at least {self.config.outlier_neighbors + 1} are needed for filtering."
            )

        filtered, _ = cloud.remove_statistical_outlier(
            nb_neighbors=self.config.outlier_neighbors,
            std_ratio=self.config.outlier_std_ratio,
        )
        if filtered.is_empty():
            raise PipelineError("Statistical outlier removal discarded every point.")
        if self.config.voxel_size > 0:
            filtered = filtered.voxel_down_sample(self.config.voxel_size)
        if len(filtered.points) < 4:
            raise PipelineError("Too few points remain to reconstruct a surface.")
        return filtered

    def reconstruct_mesh(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """Estimate normals and build a mesh using Poisson or ball pivoting."""
        if not cloud.has_normals():
            extent = float(np.linalg.norm(cloud.get_max_bound() - cloud.get_min_bound()))
            if not np.isfinite(extent) or extent <= 0:
                raise PipelineError("Point cloud has invalid or zero spatial extent.")
            radius = max(extent / max(len(cloud.points) ** (1 / 3), 1), extent * 0.005)
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
            )
            cloud.orient_normals_consistent_tangent_plane(
                min(50, max(3, len(cloud.points) - 1))
            )
        if not cloud.has_normals():
            raise PipelineError("Could not estimate point-cloud normals.")

        if self.config.reconstruction_method == "poisson":
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cloud, depth=self.config.poisson_depth
            )
            density_values = np.asarray(densities)
            if len(density_values):
                threshold = np.quantile(density_values, self.config.poisson_trim_fraction)
                mesh.remove_vertices_by_mask(density_values < threshold)
        else:
            distances = np.asarray(cloud.compute_nearest_neighbor_distance())
            distances = distances[np.isfinite(distances) & (distances > 0)]
            if not len(distances):
                raise PipelineError("Cannot estimate ball-pivoting radii from this point cloud.")
            spacing = float(np.median(distances))
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                cloud,
                o3d.utility.DoubleVector([spacing * 1.5, spacing * 3.0, spacing * 5.0]),
            )

        if mesh.is_empty() or len(mesh.triangles) == 0:
            raise PipelineError(
                f"{self.config.reconstruction_method} reconstruction produced no triangles."
            )
        return mesh


class MeshOptimizer:
    """Repair, decimate, and export a mesh within configured limits."""

    def __init__(self, config: PipelineConfig) -> None:
        """Keep the shared pipeline settings."""
        _require_dependencies(require_open3d=True, require_trimesh=True)
        self.config = config

    def optimize_and_export(self, mesh: o3d.geometry.TriangleMesh) -> tuple[Path, Path]:
        """Repair a mesh, enforce face/file-size limits, and write STL and OBJ."""
        repaired = self._repair(mesh)
        original_faces = len(repaired.triangles)
        target_faces = min(original_faces, self.config.max_faces)
        max_bytes = int(self.config.max_file_size_mb * 1_000_000)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        stl_path = self.config.output_dir / "model.stl"
        obj_path = self.config.output_dir / "model.obj"

        while True:
            candidate = repaired
            if target_faces < original_faces:
                candidate = repaired.simplify_quadric_decimation(
                    target_number_of_triangles=target_faces
                )
                if candidate.is_empty() or len(candidate.triangles) == 0:
                    raise PipelineError("Quadratic decimation produced an empty mesh.")
            stl_data = self._export_bytes(candidate, "stl")
            obj_data = self._export_bytes(candidate, "obj")
            if len(stl_data) <= max_bytes and len(obj_data) <= max_bytes:
                stl_path.write_bytes(stl_data)
                obj_path.write_bytes(obj_data)
                LOGGER.info(
                    "Exported %d triangles (STL %.2f MB, OBJ %.2f MB)",
                    len(candidate.triangles),
                    len(stl_data) / 1_000_000,
                    len(obj_data) / 1_000_000,
                )
                return stl_path, obj_path
            if target_faces <= self.config.min_decimation_faces:
                raise PipelineError(
                    "Could not meet the configured file-size limit even at the minimum "
                    f"face count ({self.config.min_decimation_faces})."
                )
            size_ratio = max(len(stl_data), len(obj_data)) / max_bytes
            target_faces = max(
                self.config.min_decimation_faces,
                min(target_faces - 1, int(target_faces / max(size_ratio * 1.05, 1.1))),
            )

    @staticmethod
    def _repair(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Remove invalid geometry, fill simple holes, and orient normals."""
        result = o3d.geometry.TriangleMesh(mesh)
        result.remove_degenerate_triangles()
        result.remove_duplicated_triangles()
        result.remove_duplicated_vertices()
        result.remove_unreferenced_vertices()
        result.remove_non_manifold_edges()
        if result.is_empty() or len(result.triangles) == 0:
            raise PipelineError("Mesh cleanup removed all triangles.")

        trimesh_mesh = trimesh.Trimesh(
            vertices=np.asarray(result.vertices),
            faces=np.asarray(result.triangles),
            process=True,
        )
        if not trimesh_mesh.is_winding_consistent:
            trimesh.repair.fix_winding(trimesh_mesh)
        trimesh.repair.fix_normals(trimesh_mesh, multibody=True)
        trimesh.repair.fill_holes(trimesh_mesh)
        result = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(trimesh_mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(trimesh_mesh.faces, dtype=np.int32)),
        )
        result.remove_degenerate_triangles()
        result.remove_duplicated_triangles()
        result.remove_duplicated_vertices()
        result.remove_unreferenced_vertices()
        if result.is_empty() or len(result.triangles) == 0:
            raise PipelineError("Mesh repair produced no usable triangles.")
        if not result.orient_triangles():
            LOGGER.warning("Mesh contains disconnected or non-orientable components.")
        result.compute_vertex_normals()
        return result

    @staticmethod
    def _export_bytes(mesh: o3d.geometry.TriangleMesh, file_type: str) -> bytes:
        """Serialize an Open3D mesh with Trimesh and normalize its result to bytes."""
        converted = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.triangles),
            process=False,
        )
        data = converted.export(file_type=file_type)
        return data.encode("utf-8") if isinstance(data, str) else data


class PhotogrammetryPipeline:
    """Orchestrate COLMAP reconstruction and Open3D/Trimesh post-processing."""

    def __init__(self, config: PipelineConfig) -> None:
        """Create pipeline stages from the supplied configuration."""
        self.config = config
        self.reconstructor = ColmapReconstructor(
            executable=config.colmap_executable,
            timeout_seconds=config.command_timeout_seconds,
        )
        self.point_cloud_processor = PointCloudProcessor(config)
        self.mesh_optimizer = MeshOptimizer(config)

    def run(self) -> tuple[Path, Path]:
        """Run the full pipeline and return the generated STL and OBJ paths."""
        images = self.reconstructor._find_images(self.config.image_dir)
        if not self.config.min_images <= len(images) <= self.config.max_images:
            raise PipelineError(
                f"Expected {self.config.min_images}-{self.config.max_images} images, "
                f"found {len(images)}."
            )
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = self.config.output_dir / "work"
        cloud_path = self.reconstructor.reconstruct(
            self.config.image_dir, work_dir, use_gpu=self.config.use_gpu
        )
        cloud = self.point_cloud_processor.load_and_clean(cloud_path)
        mesh = self.point_cloud_processor.reconstruct_mesh(cloud)
        return self.mesh_optimizer.optimize_and_export(mesh)


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for running the pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="Directory containing overlapping photos")
    parser.add_argument("output_dir", type=Path, help="Directory for model.stl and model.obj")
    parser.add_argument(
        "--method", choices=("poisson", "ball_pivoting"), default="poisson",
        help="Surface reconstruction algorithm (default: poisson)",
    )
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable or full path")
    parser.add_argument("--voxel-size", type=float, default=0.0)
    parser.add_argument("--max-faces", type=int, default=250_000)
    parser.add_argument("--max-file-size-mb", type=float, default=25.0)
    parser.add_argument("--use-gpu", action="store_true", help="Enable COLMAP SIFT GPU processing")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the CLI, returning a nonzero exit code for pipeline errors."""
    args = _parse_args(arguments)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        config = PipelineConfig(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            colmap_executable=args.colmap,
            reconstruction_method=args.method,
            voxel_size=args.voxel_size,
            max_faces=args.max_faces,
            max_file_size_mb=args.max_file_size_mb,
            use_gpu=args.use_gpu,
        )
        stl_path, obj_path = PhotogrammetryPipeline(config).run()
    except (PipelineError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("Created %s and %s", stl_path, obj_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
