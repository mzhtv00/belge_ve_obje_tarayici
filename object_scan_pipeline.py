"""Photogrammetry, point-cloud cleanup, mesh reconstruction, and export pipeline.

The photogrammetry stage uses the COLMAP command-line application. Open3D is
used for point-cloud and mesh operations; Trimesh handles mesh repair/export.
Install the native COLMAP executable separately and make it available on PATH.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np


LOGGER = logging.getLogger("object_scan_pipeline")
ImageExtensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ReconstructionMethod = Literal["poisson", "ball_pivoting"]
OutputFormat = Literal["stl", "obj"]


class PipelineError(RuntimeError):
    """Base exception for expected pipeline failures."""


class InputValidationError(PipelineError):
    """Raised when an image set or pipeline configuration is invalid."""


class DependencyError(PipelineError):
    """Raised when a required Python package or executable is unavailable."""


class ExternalToolError(PipelineError):
    """Raised when COLMAP exits unsuccessfully or exceeds its timeout."""


class ReconstructionError(PipelineError):
    """Raised when a point cloud cannot be converted into a usable mesh."""


class MeshValidationError(PipelineError):
    """Raised when an optimized mesh cannot meet export requirements."""


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration. Distances are in the reconstruction's units."""

    work_dir: Path
    min_images: int = 15
    max_images: int = 36
    colmap_executable: str = "colmap"
    use_gpu: bool = True
    subprocess_timeout_seconds: int = 7200
    max_image_bytes: int = 25 * 1024 * 1024
    max_image_edge_px: int = 12000
    voxel_size: float = 0.005
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    reconstruction_method: ReconstructionMethod = "poisson"
    poisson_depth: int = 9
    poisson_density_quantile: float = 0.02
    ball_pivoting_radius_multipliers: tuple[float, ...] = (1.5, 2.5, 4.0)
    target_triangles: int = 300_000
    max_export_bytes: int = 25_000_000
    require_watertight: bool = True
    scale_to_mm: float | None = None

    def __post_init__(self) -> None:
        """Reject invalid configuration early, before expensive processing."""
        if self.min_images < 2 or self.max_images < self.min_images:
            raise InputValidationError("Image-count limits must satisfy 2 <= min <= max.")
        if self.subprocess_timeout_seconds <= 0 or self.max_image_edge_px <= 0:
            raise InputValidationError("Timeout and image-dimension limits must be positive.")
        if self.voxel_size <= 0 or not math.isfinite(self.voxel_size):
            raise InputValidationError("voxel_size must be a finite positive number.")
        if (
            self.outlier_neighbors < 3
            or self.outlier_std_ratio <= 0
            or not math.isfinite(self.outlier_std_ratio)
        ):
            raise InputValidationError("Outlier-removal parameters must be positive.")
        if self.reconstruction_method not in ("poisson", "ball_pivoting"):
            raise InputValidationError("reconstruction_method must be poisson or ball_pivoting.")
        if not 4 <= self.poisson_depth <= 14:
            raise InputValidationError("poisson_depth must be between 4 and 14.")
        if not 0 <= self.poisson_density_quantile < 0.5:
            raise InputValidationError("poisson_density_quantile must be in [0, 0.5).")
        if self.target_triangles < 100:
            raise InputValidationError("target_triangles must be at least 100.")
        if self.max_export_bytes <= 0 or self.max_image_bytes <= 0:
            raise InputValidationError("File-size limits must be positive.")
        if self.scale_to_mm is not None and (
            self.scale_to_mm <= 0 or not math.isfinite(self.scale_to_mm)
        ):
            raise InputValidationError("scale_to_mm must be finite and positive when set.")
        if not self.ball_pivoting_radius_multipliers or any(
            radius <= 0 or not math.isfinite(radius)
            for radius in self.ball_pivoting_radius_multipliers
        ):
            raise InputValidationError("Ball-pivoting radius multipliers must be positive.")


@dataclass(frozen=True)
class ImageSet:
    """Validated, deterministically ordered images used by one scan."""

    paths: tuple[Path, ...]

    @classmethod
    def from_directory(
        cls, image_dir: Path, config: PipelineConfig
    ) -> ImageSet:
        """Read and validate the supported images in a directory."""
        root = image_dir.expanduser().resolve()
        if not root.is_dir():
            raise InputValidationError(f"Image directory does not exist: {root}")

        paths = tuple(
            sorted(
                (path for path in root.iterdir() if path.is_file()
                 and path.suffix.lower() in ImageExtensions),
                key=lambda path: path.name.casefold(),
            )
        )
        if not config.min_images <= len(paths) <= config.max_images:
            raise InputValidationError(
                f"Expected {config.min_images}-{config.max_images} images; found {len(paths)}."
            )
        folded_names = [path.name.casefold() for path in paths]
        if len(folded_names) != len(set(folded_names)):
            raise InputValidationError(
                "Image filenames must be unique without case sensitivity."
            )
        for path in paths:
            if path.stat().st_size == 0:
                raise InputValidationError(f"Empty image file: {path.name}")
            if path.stat().st_size > config.max_image_bytes:
                raise InputValidationError(f"Image exceeds size limit: {path.name}")
            cv2 = _require_package("cv2")
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise InputValidationError(f"Image is corrupt or unsupported: {path.name}")
            height, width = image.shape[:2]
            if max(height, width) > config.max_image_edge_px:
                raise InputValidationError(
                    f"Image edge exceeds configured limit: {path.name}"
                )
        return cls(paths=paths)

    def stage(self, destination: Path) -> Path:
        """Copy only validated inputs into an isolated COLMAP image directory."""
        destination.mkdir(parents=True, exist_ok=True)
        try:
            for source in self.paths:
                shutil.copy2(source, destination / source.name)
        except OSError as exc:
            raise PipelineError(f"Could not stage scan images: {exc}") from exc
        return destination


@dataclass(frozen=True)
class ColmapArtifacts:
    """Paths produced by a successful COLMAP sparse and dense run."""

    sparse_model: Path
    dense_workspace: Path
    fused_point_cloud: Path


@dataclass(frozen=True)
class ExportArtifact:
    """Metadata for one exported mesh file."""

    path: Path
    format: OutputFormat
    triangle_count: int
    byte_count: int
    watertight: bool
    scale_to_mm: float | None


@dataclass(frozen=True)
class PipelineResult:
    """Files and work directory produced by a completed scan."""

    run_dir: Path
    cleaned_point_cloud: Path
    exports: tuple[ExportArtifact, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _require_package(package_name: str) -> Any:
    """Import an optional heavy dependency and report an actionable error."""
    try:
        return importlib.import_module(package_name)
    except ImportError as exc:
        raise DependencyError(
            f"Missing Python dependency '{package_name}'. Install the pipeline requirements."
        ) from exc


class ColmapPhotogrammetry:
    """Run SIFT feature extraction, matching, SfM, and dense fusion with COLMAP."""

    def __init__(self, config: PipelineConfig) -> None:
        """Create a runner using the configured COLMAP executable and limits."""
        self._config = config
        executable = shutil.which(config.colmap_executable)
        if executable is None:
            raise DependencyError(
                f"COLMAP executable '{config.colmap_executable}' was not found on PATH."
            )
        self._executable = executable

    def reconstruct(self, images: ImageSet, run_dir: Path) -> ColmapArtifacts:
        """Produce a fused dense PLY point cloud from overlapping photos."""
        workspace = run_dir / "colmap"
        database = workspace / "database.db"
        sparse_dir = workspace / "sparse"
        dense_dir = workspace / "dense"
        image_dir = images.stage(workspace / "images")
        workspace.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        self._run(
            "feature_extractor",
            [
                "--database_path", str(database),
                "--image_path", str(image_dir),
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu", self._bool_arg(self._config.use_gpu),
                "--SiftExtraction.max_num_features", "8192",
            ],
            workspace,
        )
        self._run(
            "exhaustive_matcher",
            [
                "--database_path", str(database),
                "--SiftMatching.use_gpu", self._bool_arg(self._config.use_gpu),
                "--SiftMatching.guided_matching", "1",
            ],
            workspace,
        )
        self._run(
            "mapper",
            [
                "--database_path", str(database),
                "--image_path", str(image_dir),
                "--output_path", str(sparse_dir),
            ],
            workspace,
        )
        sparse_models = sorted(
            (path for path in sparse_dir.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        )
        if not sparse_models:
            raise ReconstructionError(
                "COLMAP registered no camera model. Check image overlap, focus, and texture."
            )
        sparse_model = sparse_models[0]

        dense_dir.mkdir(parents=True, exist_ok=True)
        self._run(
            "image_undistorter",
            [
                "--image_path", str(image_dir),
                "--input_path", str(sparse_model),
                "--output_path", str(dense_dir),
                "--output_type", "COLMAP",
            ],
            workspace,
        )
        self._run(
            "patch_match_stereo",
            [
                "--workspace_path", str(dense_dir),
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.geom_consistency", "true",
            ],
            workspace,
        )
        fused_cloud = dense_dir / "fused.ply"
        self._run(
            "stereo_fusion",
            [
                "--workspace_path", str(dense_dir),
                "--workspace_format", "COLMAP",
                "--input_type", "geometric",
                "--output_path", str(fused_cloud),
            ],
            workspace,
        )
        if not fused_cloud.is_file() or fused_cloud.stat().st_size == 0:
            raise ReconstructionError("COLMAP completed without producing fused.ply.")
        return ColmapArtifacts(sparse_model, dense_dir, fused_cloud)

    @staticmethod
    def _bool_arg(value: bool) -> str:
        """Return the integer boolean format accepted by COLMAP options."""
        return "1" if value else "0"

    def _run(self, command: str, arguments: Sequence[str], cwd: Path) -> None:
        """Run one COLMAP command without a shell and persist its diagnostic log."""
        argv = [self._executable, command, *arguments]
        log_path = cwd / f"{command}.log"
        LOGGER.info("Running COLMAP command: %s", command)
        try:
            result = subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self._config.subprocess_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExternalToolError(
                f"COLMAP '{command}' exceeded "
                f"{self._config.subprocess_timeout_seconds} seconds."
            ) from exc
        except OSError as exc:
            raise ExternalToolError(f"Could not start COLMAP '{command}': {exc}") from exc

        log_path.write_text(
            f"$ {' '.join(argv)}\n\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            encoding="utf-8",
        )
        if result.returncode != 0:
            diagnostic = (result.stderr or result.stdout)[-4000:]
            raise ExternalToolError(
                f"COLMAP '{command}' failed with exit code {result.returncode}. "
                f"See {log_path}.\n{diagnostic}"
            )


class PointCloudProcessor:
    """Clean a dense cloud and estimate/orient normals for meshing."""

    def __init__(self, config: PipelineConfig) -> None:
        """Store point-cloud filtering parameters."""
        self._config = config

    def process(self, input_path: Path, output_path: Path) -> Any:
        """Downsample, remove statistical outliers, and save a normal-bearing PLY."""
        o3d = _require_package("open3d")
        cloud = o3d.io.read_point_cloud(str(input_path))
        if cloud.is_empty():
            raise ReconstructionError(f"Point cloud is empty or unreadable: {input_path}")

        cloud = cloud.voxel_down_sample(voxel_size=self._config.voxel_size)
        if len(cloud.points) < 100:
            raise ReconstructionError(
                "Too few points remain after voxel downsampling; adjust voxel_size."
            )
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=min(self._config.outlier_neighbors, len(cloud.points) - 1),
            std_ratio=self._config.outlier_std_ratio,
        )
        if cloud.is_empty() or len(cloud.points) < 100:
            raise ReconstructionError(
                "Outlier removal left too few points to reconstruct a surface."
            )

        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self._config.voxel_size * 3.0,
                max_nn=max(30, self._config.outlier_neighbors),
            )
        )
        cloud.orient_normals_consistent_tangent_plane(
            min(30, max(3, len(cloud.points) - 1))
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(output_path), cloud, write_ascii=False):
            raise PipelineError(f"Could not write cleaned point cloud: {output_path}")
        return cloud


class SurfaceReconstructor:
    """Build a triangle surface using Poisson or Ball Pivoting in Open3D."""

    def __init__(self, config: PipelineConfig) -> None:
        """Store surface reconstruction and filtering settings."""
        self._config = config

    def reconstruct(self, cloud: Any) -> Any:
        """Return a cleaned triangle mesh, trimming low-density Poisson regions."""
        o3d = _require_package("open3d")
        if not cloud.has_normals():
            raise ReconstructionError("Surface reconstruction requires oriented normals.")

        if self._config.reconstruction_method == "poisson":
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cloud, depth=self._config.poisson_depth
            )
            density_values = np.asarray(densities)
            threshold = np.quantile(
                density_values, self._config.poisson_density_quantile
            )
            mesh.remove_vertices_by_mask(density_values < threshold)
        else:
            radii = [
                self._config.voxel_size * multiplier
                for multiplier in self._config.ball_pivoting_radius_multipliers
            ]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                cloud, o3d.utility.DoubleVector(radii)
            )

        if mesh.is_empty() or len(mesh.triangles) < 100:
            raise ReconstructionError(
                "Surface reconstruction produced no usable mesh. "
                "Try another method or improve image overlap and point-cloud density."
            )
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        if mesh.is_empty() or len(mesh.triangles) < 100:
            raise ReconstructionError("Mesh cleanup removed too many triangles.")
        return mesh


class MeshOptimizer:
    """Repair, decimate, validate, and export a mesh for CAD/slicer workflows."""

    def __init__(self, config: PipelineConfig) -> None:
        """Store export limits and strict manifold validation behavior."""
        self._config = config

    def optimize_and_export(
        self,
        mesh: Any,
        output_dir: Path,
        formats: Sequence[OutputFormat] = ("stl", "obj"),
    ) -> tuple[ExportArtifact, ...]:
        """Repair and decimate a mesh until all requested files meet size limits."""
        trimesh = _require_package("trimesh")
        requested = tuple(dict.fromkeys(formats))
        if not requested or any(fmt not in ("stl", "obj") for fmt in requested):
            raise InputValidationError("formats must contain stl and/or obj.")
        output_dir.mkdir(parents=True, exist_ok=True)

        cleaned = mesh
        cleaned.remove_degenerate_triangles()
        cleaned.remove_duplicated_triangles()
        cleaned.remove_duplicated_vertices()
        cleaned.remove_unreferenced_vertices()
        if self._config.scale_to_mm is not None:
            cleaned.scale(self._config.scale_to_mm, center=cleaned.get_center())
        cleaned.compute_vertex_normals()

        current_target = min(self._config.target_triangles, len(cleaned.triangles))
        for _ in range(8):
            candidate = cleaned
            if len(cleaned.triangles) > current_target:
                candidate = cleaned.simplify_quadric_decimation(
                    target_number_of_triangles=current_target
                )
                candidate.remove_degenerate_triangles()
                candidate.remove_duplicated_triangles()
                candidate.remove_duplicated_vertices()
                candidate.remove_unreferenced_vertices()
            if candidate.is_empty() or len(candidate.triangles) < 4:
                raise MeshValidationError("Decimation produced an unusable mesh.")

            repaired = self._repair(candidate, trimesh)
            if self._config.require_watertight and not repaired.is_watertight:
                raise MeshValidationError(
                    "Mesh is not watertight after hole filling. "
                    "Inspect the reconstruction or enable a dedicated mesh-repair tool."
                )
            payloads = {fmt: self._serialize(repaired, fmt) for fmt in requested}
            if all(len(data) <= self._config.max_export_bytes for data in payloads.values()):
                artifacts = self._write_artifacts(
                    payloads, requested, output_dir, len(repaired.faces), repaired.is_watertight
                )
                return artifacts
            if current_target <= 100:
                break
            current_target = max(100, int(current_target * 0.75))

        raise MeshValidationError(
            f"Could not fit requested exports under {self._config.max_export_bytes} bytes. "
            "Lower target_triangles or export fewer formats."
        )

    def _repair(self, mesh: Any, trimesh_module: Any) -> Any:
        """Fill small holes where possible and normalize face winding/normals."""
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        repaired = trimesh_module.Trimesh(
            vertices=vertices, faces=faces, process=True, validate=True
        )
        repaired.remove_unreferenced_vertices()
        trimesh_module.repair.fill_holes(repaired)
        trimesh_module.repair.fix_winding(repaired)
        trimesh_module.repair.fix_normals(repaired, multibody=True)
        if repaired.is_empty or not repaired.is_winding_consistent:
            raise MeshValidationError("Mesh has invalid or inconsistent face winding.")
        if not repaired.is_watertight and self._config.require_watertight:
            raise MeshValidationError(
                "Hole filling could not close every boundary. "
                "Trimesh fills small holes; larger boundaries need review."
            )
        return repaired

    def _serialize(self, mesh: Any, fmt: OutputFormat) -> bytes:
        """Serialize one mesh format into bytes for size checking."""
        try:
            data = mesh.export(file_type=fmt)
        except (ValueError, TypeError) as exc:
            raise MeshValidationError(f"Could not serialize {fmt.upper()} mesh: {exc}") from exc
        return data.encode("utf-8") if isinstance(data, str) else bytes(data)

    def _write_artifacts(
        self,
        payloads: dict[OutputFormat, bytes],
        formats: tuple[OutputFormat, ...],
        output_dir: Path,
        triangle_count: int,
        watertight: bool,
    ) -> tuple[ExportArtifact, ...]:
        """Atomically write completed exports and return their metadata."""
        artifacts: list[ExportArtifact] = []
        staged: list[tuple[Path, Path, OutputFormat]] = []
        try:
            for fmt in formats:
                target = output_dir / f"model.{fmt}"
                temporary = output_dir / f".model.{uuid.uuid4().hex}.{fmt}.tmp"
                temporary.write_bytes(payloads[fmt])
                staged.append((temporary, target, fmt))
            for temporary, target, fmt in staged:
                temporary.replace(target)
                artifacts.append(
                    ExportArtifact(
                        path=target,
                        format=fmt,
                        triangle_count=triangle_count,
                        byte_count=target.stat().st_size,
                        watertight=watertight,
                        scale_to_mm=self._config.scale_to_mm,
                    )
                )
        except OSError as exc:
            for temporary, _, _ in staged:
                temporary.unlink(missing_ok=True)
            raise PipelineError(f"Could not write mesh export: {exc}") from exc
        return tuple(artifacts)


class ObjectScanPipeline:
    """Orchestrate the complete image-to-mesh scan workflow."""

    def __init__(self, config: PipelineConfig) -> None:
        """Wire photogrammetry and geometry services using one configuration."""
        self._config = config
        self._photogrammetry = ColmapPhotogrammetry(config)
        self._point_cloud_processor = PointCloudProcessor(config)
        self._surface_reconstructor = SurfaceReconstructor(config)
        self._mesh_optimizer = MeshOptimizer(config)

    def run(
        self,
        image_dir: Path,
        output_dir: Path,
        formats: Sequence[OutputFormat] = ("stl", "obj"),
    ) -> PipelineResult:
        """Run feature matching through export and return persistent artifact paths."""
        started = time.monotonic()
        images = ImageSet.from_directory(image_dir, self._config)
        run_dir = self._config.work_dir.expanduser().resolve() / f"scan-{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=False)
        LOGGER.info("Starting scan with %d images in %s", len(images.paths), run_dir)

        colmap_result = self._photogrammetry.reconstruct(images, run_dir)
        cleaned_cloud_path = run_dir / "pointcloud" / "cleaned.ply"
        cloud = self._point_cloud_processor.process(
            colmap_result.fused_point_cloud, cleaned_cloud_path
        )
        raw_mesh = self._surface_reconstructor.reconstruct(cloud)
        exports = self._mesh_optimizer.optimize_and_export(
            raw_mesh, output_dir.expanduser().resolve(), formats
        )
        LOGGER.info(
            "Scan completed in %.1f seconds; exported %s",
            time.monotonic() - started,
            ", ".join(str(item.path) for item in exports),
        )
        return PipelineResult(
            run_dir=run_dir,
            cleaned_point_cloud=cleaned_cloud_path,
            exports=exports,
        )


def _build_parser() -> argparse.ArgumentParser:
    """Build the small command-line entry point for running the pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory with 15-36 photos.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for STL/OBJ files.")
    parser.add_argument("--work-dir", type=Path, default=Path("./scan-work"))
    parser.add_argument("--method", choices=("poisson", "ball_pivoting"), default="poisson")
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--target-triangles", type=int, default=300_000)
    parser.add_argument("--scale-to-mm", type=float, default=None)
    parser.add_argument(
        "--cpu-features",
        action="store_true",
        help="Disable GPU for SIFT extraction/matching; dense stereo still needs CUDA COLMAP.",
    )
    parser.add_argument(
        "--allow-open-mesh",
        action="store_true",
        help="Permit non-watertight mesh export (not recommended for 3D printing).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and report expected failures cleanly."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = PipelineConfig(
            work_dir=args.work_dir,
            use_gpu=not args.cpu_features,
            reconstruction_method=args.method,
            voxel_size=args.voxel_size,
            target_triangles=args.target_triangles,
            scale_to_mm=args.scale_to_mm,
            require_watertight=not args.allow_open_mesh,
        )
        result = ObjectScanPipeline(config).run(args.images, args.output)
    except PipelineError as exc:
        LOGGER.error("%s", exc)
        return 2
    for artifact in result.exports:
        print(
            f"{artifact.path} | {artifact.triangle_count} triangles | "
            f"{artifact.byte_count} bytes | watertight={artifact.watertight}"
        )
    print(f"Work files: {result.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
