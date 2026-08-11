#!/usr/bin/env python3
"""Self-contained 3DGS rendering backend for the localization server.

The benchmark scripts in this repository render through a private fork of the
3DGS codebase (``Gaussian_splatting.scene.SceneRender`` /
``VirtualCamera2``). That fork is not part of this checkout, and ``VirtualCamera2``
only carries a field of view, so it silently assumes the principal point sits at
the image centre. Real calibrations do not: ``data/colmap_E2`` has
``cx = 634.7`` against ``width / 2 = 612``, a 22 px shift that would bias every
PnP solution.

This module renders with the stock ``gaussian-splatting`` checkout instead, using
a camera that carries full pinhole intrinsics (fx, fy, cx, cy) in an off-centre
projection matrix. It is the same construction as
``fast_3dgs_renderer/render_pose_with_depth.py``.

Depth convention
----------------
The CUDA rasterizer accumulates ``expected_invdepth += (1 / z) * alpha * T``
(``cuda_rasterizer/forward.cu``) and its Python wrapper returns that tensor, which
``gaussian_renderer.render()`` then exposes under the key ``"depth"``. It is an
*inverse* depth map, so it must be reciprocated before it can be used to
backproject pixels. ``depth_mode="inverse"`` (the default) does that;
``depth_mode="linear"`` is for rasterizer builds that return metric z directly.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

DEFAULT_GS_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gaussian-splatting")


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def import_gs_modules(repo_path: str = DEFAULT_GS_REPO):
    """Import the 3DGS render entry points from a 3DGS repository root.

    3DGS uses top-level absolute imports (``from scene.gaussian_model import
    GaussianModel``, ``from utils.sh_utils import eval_sh``), so its root has to
    be on ``sys.path``. Putting it merely at the front is not enough: this
    repository ships a *regular* top-level ``utils`` package (with an
    ``__init__.py``) while ``gaussian-splatting/utils`` has none, so it is only a
    namespace portion. Python keeps scanning past a namespace portion and a
    regular package found later wins, which makes ``utils.system_utils``
    unresolvable.

    So this repository's directory is removed from ``sys.path`` for the duration
    of the import. Afterwards the modules are cached, and ``sys.path`` is put
    back as it was.

    One consequence: for the rest of the process, ``import utils`` refers to
    3DGS' ``utils``, not this repository's. Nothing on the server path uses the
    latter (only the legacy benchmark scripts do).
    """
    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(os.path.join(repo_path, "gaussian_renderer")):
        raise FileNotFoundError(
            "Not a 3DGS repository (no gaussian_renderer/): {}".format(repo_path)
        )

    saved_path = list(sys.path)
    shadowing = {_THIS_DIR, os.path.abspath(os.getcwd()), ""}
    try:
        sys.path = [repo_path] + [
            entry
            for entry in sys.path
            if entry not in shadowing and os.path.abspath(entry or ".") != _THIS_DIR
        ]
        from gaussian_renderer import render as gs_render
        from scene.gaussian_model import GaussianModel
    finally:
        sys.path = saved_path
    return gs_render, GaussianModel


@dataclass
class PinholeCamera:
    """Pinhole intrinsics for one image resolution."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def scaled_to(self, width: int, height: int) -> "PinholeCamera":
        """Rescale intrinsics to a different image resolution."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive")
        if int(width) == int(self.width) and int(height) == int(self.height):
            return self
        sx = float(width) / float(self.width)
        sy = float(height) / float(self.height)
        return PinholeCamera(
            width=int(width),
            height=int(height),
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "cam_width": int(self.width),
            "cam_height": int(self.height),
            "cam_fx": float(self.fx),
            "cam_fy": float(self.fy),
            "cam_cx": float(self.cx),
            "cam_cy": float(self.cy),
        }


@dataclass
class PipelineConfig:
    """The handful of fields ``gaussian_renderer.render()`` reads off ``pipe``.

    Using this instead of ``arguments.PipelineParams`` keeps the renderer free of
    argparse, which matters because roslaunch appends ``__name:=`` style
    arguments that argparse cannot parse.
    """

    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False
    antialiasing: bool = False


def focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def c2w_to_w2c(pose_c2w: np.ndarray) -> np.ndarray:
    """Invert a rigid camera-to-world transform without a general matrix inverse."""
    pose_c2w = np.asarray(pose_c2w, dtype=np.float64)
    r_wc = pose_c2w[:3, :3]
    centre = pose_c2w[:3, 3]
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = r_wc.T
    w2c[:3, 3] = -r_wc.T @ centre
    return w2c


def w2c_to_c2w(pose_w2c: np.ndarray) -> np.ndarray:
    pose_w2c = np.asarray(pose_w2c, dtype=np.float64)
    r_cw = pose_w2c[:3, :3]
    t_cw = pose_w2c[:3, 3]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = r_cw.T
    c2w[:3, 3] = -r_cw.T @ t_cw
    return c2w


class RenderCamera:
    """Minimal viewpoint object carrying the attributes ``render()`` reads.

    ``R``/``T`` follow the 3DGS convention: ``R`` is the *transpose* of the
    world-to-camera rotation and ``T`` is the world-to-camera translation.
    """

    def __init__(
        self,
        camera: PinholeCamera,
        pose_w2c: np.ndarray,
        device: str = "cuda",
        znear: float = 0.01,
        zfar: float = 1000.0,
        use_principal_point: bool = True,
        image_name: str = "render",
        uid: int = 0,
    ) -> None:
        self.uid = uid
        self.colmap_id = uid
        self.image_name = image_name
        self.image_width = int(camera.width)
        self.image_height = int(camera.height)
        self.width = self.image_width
        self.height = self.image_height
        self.znear = float(znear)
        self.zfar = float(zfar)

        self.FoVx = focal2fov(camera.fx, camera.width)
        self.FoVy = focal2fov(camera.fy, camera.height)

        pose_w2c = np.asarray(pose_w2c, dtype=np.float64)
        world_view = np.zeros((4, 4), dtype=np.float32)
        world_view[:3, :3] = pose_w2c[:3, :3]
        world_view[:3, 3] = pose_w2c[:3, 3]
        world_view[3, 3] = 1.0

        if use_principal_point:
            projection = self._projection_from_intrinsics(camera, znear, zfar)
        else:
            projection = self._projection_centred(self.FoVx, self.FoVy, znear, zfar)

        world_view_t = torch.from_numpy(world_view)
        self.world_view_transform = world_view_t.transpose(0, 1).to(device)
        self.projection_matrix = projection.transpose(0, 1).to(device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # Unused by rendering, but some variants of render() touch them.
        self.original_image = None
        self.gt_alpha_mask = None
        self.depth = None
        self.invdepthmap = None
        self.depth_params = None
        self.is_test_dataset = False
        self.is_test_view = False

    @staticmethod
    def _projection_from_intrinsics(
        camera: PinholeCamera, znear: float, zfar: float
    ) -> torch.Tensor:
        """Off-centre projection carrying the principal-point shift.

        Reduces to the stock centred matrix when cx = width/2 and cy = height/2.
        The rasterizer still derives its Jacobian from FoVx/FoVy (equivalently
        fx/fy), which stays correct; only the NDC centre moves.
        """
        p = torch.zeros(4, 4, dtype=torch.float32)
        p[0, 0] = 2.0 * float(camera.fx) / float(camera.width)
        p[1, 1] = 2.0 * float(camera.fy) / float(camera.height)
        p[0, 2] = 1.0 - 2.0 * float(camera.cx) / float(camera.width)
        p[1, 2] = 2.0 * float(camera.cy) / float(camera.height) - 1.0
        p[3, 2] = 1.0
        p[2, 2] = float(zfar) / (float(zfar) - float(znear))
        p[2, 3] = -(float(zfar) * float(znear)) / (float(zfar) - float(znear))
        return p

    @staticmethod
    def _projection_centred(
        fovx: float, fovy: float, znear: float, zfar: float
    ) -> torch.Tensor:
        tan_half_x = math.tan(fovx / 2.0)
        tan_half_y = math.tan(fovy / 2.0)
        p = torch.zeros(4, 4, dtype=torch.float32)
        p[0, 0] = 1.0 / tan_half_x
        p[1, 1] = 1.0 / tan_half_y
        p[3, 2] = 1.0
        p[2, 2] = float(zfar) / (float(zfar) - float(znear))
        p[2, 3] = -(float(zfar) * float(znear)) / (float(zfar) - float(znear))
        return p


class GaussianMap:
    """A 3DGS model held in GPU memory, renderable from arbitrary poses."""

    def __init__(
        self,
        model_path: str,
        iteration: int = -1,
        sh_degree: int = 3,
        white_background: bool = False,
        device: str = "cuda",
        depth_mode: str = "inverse",
        znear: float = 0.01,
        zfar: float = 1000.0,
        use_principal_point: bool = True,
        gs_repo_path: str = DEFAULT_GS_REPO,
        pipeline: Optional[PipelineConfig] = None,
        logger=None,
    ) -> None:
        if depth_mode not in ("inverse", "linear"):
            raise ValueError("depth_mode must be 'inverse' or 'linear'")

        self.model_path = os.path.abspath(model_path)
        self.device = device
        self.depth_mode = depth_mode
        self.znear = float(znear)
        self.zfar = float(zfar)
        self.use_principal_point = bool(use_principal_point)
        self.pipeline = pipeline or PipelineConfig()
        self._log = logger or (lambda *a: None)

        gs_render, GaussianModel = import_gs_modules(gs_repo_path)
        self._render_fn = gs_render

        self.iteration = self._resolve_iteration(self.model_path, iteration)
        ply_path = os.path.join(
            self.model_path, "point_cloud", "iteration_{}".format(self.iteration), "point_cloud.ply"
        )
        if not os.path.isfile(ply_path):
            raise FileNotFoundError("3DGS point cloud not found: {}".format(ply_path))

        self._log("Loading 3DGS point cloud: %s", ply_path)
        self.gaussians = GaussianModel(sh_degree)
        try:
            self.gaussians.load_ply(ply_path, False)
        except TypeError:  # older forks take only the path
            self.gaussians.load_ply(ply_path)

        background = [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0]
        self.background = torch.tensor(background, dtype=torch.float32, device=device)
        self._log("3DGS model ready: %d gaussians", int(self.gaussians.get_xyz.shape[0]))

    @staticmethod
    def _resolve_iteration(model_path: str, iteration: int) -> int:
        folder = os.path.join(model_path, "point_cloud")
        if not os.path.isdir(folder):
            raise FileNotFoundError("No point_cloud/ under {}".format(model_path))
        if iteration is not None and int(iteration) >= 0:
            return int(iteration)
        found = []
        for name in os.listdir(folder):
            if name.startswith("iteration_"):
                try:
                    found.append(int(name.split("_")[-1]))
                except ValueError:
                    continue
        if not found:
            raise FileNotFoundError("No iteration_* folders under {}".format(folder))
        return max(found)

    @torch.no_grad()
    def render(
        self, pose_c2w: np.ndarray, camera: PinholeCamera
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Render RGB and metric depth from a camera-to-world pose.

        Returns a float32 CHW tensor in [0, 1] on ``self.device``, and an
        ``HxW`` float32 depth map in metres where invalid pixels are NaN.
        """
        view = RenderCamera(
            camera=camera,
            pose_w2c=c2w_to_w2c(pose_c2w),
            device=self.device,
            znear=self.znear,
            zfar=self.zfar,
            use_principal_point=self.use_principal_point,
        )
        out = self._render_fn(view, self.gaussians, self.pipeline, self.background)
        rgb = out["render"].clamp(0.0, 1.0)
        raw = out["depth"].detach().float().cpu().numpy()
        raw = np.squeeze(raw)
        if raw.ndim != 2:
            raise RuntimeError("Expected a 2D depth map, got shape {}".format(raw.shape))
        return rgb, self._to_metric_depth(raw)

    def _to_metric_depth(self, raw: np.ndarray) -> np.ndarray:
        """Convert the rasterizer's depth output into metres."""
        depth = np.full(raw.shape, np.nan, dtype=np.float32)
        if self.depth_mode == "inverse":
            # Pixels never covered by a gaussian accumulate 0 inverse depth.
            valid = np.isfinite(raw) & (raw > 1e-8)
            depth[valid] = 1.0 / raw[valid]
        else:
            valid = np.isfinite(raw) & (raw > 0.0)
            depth[valid] = raw[valid]
        return depth


def load_reference_poses_from_colmap(
    path: str,
) -> Tuple[Dict[str, np.ndarray], PinholeCamera, str]:
    """Read reference poses from a COLMAP sparse model.

    ``path`` may be the model directory itself or any root that contains one in
    a usual place (``sparse/0`` and friends). The text encoding wins over the
    binary one when both are present.

    This is the preferred source: COLMAP is what the 3DGS map was trained from,
    so its world frame *is* the map frame, and unlike ``cameras.json`` it also
    carries the principal point and the distortion model.

    Returns camera-to-world poses keyed by image name, the intrinsics of the
    model's dominant camera, and the directory the model was read from.
    """
    from colmap_model import find_model_dir, read_model

    found = find_model_dir(path)
    if found is None:
        raise FileNotFoundError(
            "No COLMAP model found under {} (looked for cameras.txt/images.txt or "
            "cameras.bin/images.bin in ., sparse/0, sparse, colmap/sparse/0)".format(path)
        )
    model = read_model(found)
    colmap_camera = model.dominant_camera()
    fx, fy, cx, cy = colmap_camera.pinhole()
    camera = PinholeCamera(
        width=int(colmap_camera.width),
        height=int(colmap_camera.height),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )
    return model.poses_c2w(), camera, found


def load_reference_poses_from_cameras_json(
    model_path: str,
) -> Tuple[Dict[str, np.ndarray], Optional[PinholeCamera]]:
    """Read reference camera poses from a trained 3DGS model's ``cameras.json``.

    ``camera_to_JSON`` in the 3DGS repo writes ``position`` as the camera centre
    in world coordinates and ``rotation`` as the camera-to-world rotation, so the
    entries are already c2w. ``cameras.json`` carries no principal point, so the
    returned intrinsics assume a centred one; that only affects database renders
    used for appearance retrieval, never the localization solve.
    """
    path = os.path.join(model_path, "cameras.json")
    if not os.path.isfile(path):
        raise FileNotFoundError("cameras.json not found: {}".format(path))
    with open(path, "r") as handle:
        entries = json.load(handle)
    if not entries:
        raise RuntimeError("cameras.json is empty: {}".format(path))

    poses: Dict[str, np.ndarray] = {}
    for entry in entries:
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = np.asarray(entry["rotation"], dtype=np.float64)
        c2w[:3, 3] = np.asarray(entry["position"], dtype=np.float64)
        poses[str(entry["img_name"])] = c2w

    first = entries[0]
    camera = PinholeCamera(
        width=int(first["width"]),
        height=int(first["height"]),
        fx=float(first["fx"]),
        fy=float(first["fy"]),
        cx=float(first["width"]) / 2.0,
        cy=float(first["height"]) / 2.0,
    )
    return poses, camera


def load_reference_poses_from_loam(database_path: str) -> Tuple[Dict[str, np.ndarray], None]:
    """Read reference poses through the fork's LOAM reader (original behaviour).

    Needs the private ``Gaussian_splatting`` fork, which provides
    ``readLOAMCameras``. Returns camera-to-world poses so both backends agree.
    """
    try:
        from Gaussian_splatting.scene.dataset_readers import readLOAMCameras
    except ImportError:
        try:
            from Gaussian_splatting_old.scene.dataset_readers import readLOAMCameras
        except ImportError as exc:
            raise ImportError(
                "reference_source='loam' needs the private Gaussian_splatting fork "
                "(readLOAMCameras) on PYTHONPATH; use reference_source='cameras_json' "
                "for a stock 3DGS model directory."
            ) from exc

    cameras = readLOAMCameras(
        extrinsics_PATH=os.path.join(database_path, "loam", "0", "poses.csv"),
        intrinsics_PATH=os.path.join(database_path, "loam", "0", "camera.yaml"),
        images_folder=os.path.join(database_path, "images"),
    )
    poses: Dict[str, np.ndarray] = {}
    for camera in cameras:
        # The reader returns the 3DGS convention: R = R_w2c.T, T = t_w2c.
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = np.asarray(camera.R, dtype=np.float64).T
        w2c[:3, 3] = np.asarray(camera.T, dtype=np.float64).reshape(3)
        poses[str(camera.image_name) + ".png"] = w2c_to_c2w(w2c)
    if not poses:
        raise RuntimeError("No reference poses read from {}".format(database_path))
    return poses, None


def resolve_reference_name(name: str, poses: Dict[str, np.ndarray]) -> str:
    """Match a retrieval result against the reference-pose keys.

    Retrieval can return a bare basename, a relative path, or a name whose
    extension differs from the one the poses were keyed with.
    """
    name = str(name).strip()
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    candidates: List[str] = [name, base, stem]
    for extension in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"):
        candidates.append(stem + extension)
    for candidate in candidates:
        if candidate in poses:
            return candidate
    raise KeyError(
        "No reference pose for retrieval result '{}' (tried {})".format(name, candidates[:6])
    )
