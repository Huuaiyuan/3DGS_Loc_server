#!/usr/bin/env python3
"""The relocalization pipeline, free of ROS.

    query image
      -> NetVLAD retrieval -> nearest reference pose
      -> render RGB + depth from the 3DGS map at that pose
      -> MASt3R dense matching between render and query
      -> backproject matched render pixels through the rendered depth
      -> PnP-RANSAC -> camera pose

Only the first stage is negotiable: the initial pose can equally come from a VIO
stream (``use_vio``, see ``vio_prior.py``) or from the previous fix
(``reuse_last_pose``), with retrieval as the bootstrap and the fallback.

Keeping this separate from the node means it can be exercised offline
(``selftest_localization.py``) and reused from ROS2 or a plain script.

All poses crossing this module's boundary are camera-to-world (c2w): the
translation is the camera centre in map coordinates.
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera_config import CameraConfig
from gs_render_backend import (
    DEFAULT_GS_REPO,
    GaussianMap,
    PinholeCamera,
    PipelineConfig,
    load_reference_poses_from_cameras_json,
    load_reference_poses_from_colmap,
    load_reference_poses_from_loam,
    resolve_reference_name,
)
from mast3r_matching import match_pair, tensor_to_rgb_uint8
from netvlad_retrieval import (
    NetVLADRetrieval,
    build_database_from_images,
    build_database_from_map,
    find_reference_image_dir,
    resolve_database_descriptors,
)
from vio_prior import VioAlignment

MAST3R_DEFAULT = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


@dataclass
class LocalizationResult:
    success: bool = False
    pose_c2w: Optional[np.ndarray] = None
    candidate_image: str = ""
    # Where the initial pose came from: "vio", "previous_pose" or "retrieval".
    init_source: str = ""
    # How far the solved pose ended up from the VIO prediction, when there was
    # one. A useful health signal: it grows as the alignment goes stale.
    vio_prediction_error_m: float = float("nan")
    num_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_rmse_px: float = float("nan")
    processing_time_ms: float = 0.0
    timings_ms: Dict[str, float] = field(default_factory=dict)
    error: str = ""


def _noop(*_args, **_kwargs) -> None:
    return None


class LocalizationEngine:
    """Holds the 3DGS map, the MASt3R model and the retrieval database."""

    def __init__(
        self,
        model_path: str,
        camera_config: CameraConfig,
        runtime_dir: str,
        iteration: int = -1,
        database_path: str = "",
        reference_source: str = "auto",
        colmap_path: str = "",
        gs_repo_path: str = DEFAULT_GS_REPO,
        mast3r_model: str = MAST3R_DEFAULT,
        device: str = "cuda",
        sh_degree: int = 3,
        white_background: bool = False,
        depth_mode: str = "inverse",
        render_use_principal_point: bool = False,
        enable_retrieval: bool = True,
        retrieval_source: str = "auto",
        reference_images_dir: str = "",
        retrieval_exclude: Optional[List[str]] = None,
        retrieval_cache_dir: str = "",
        rebuild_retrieval_db: bool = False,
        db_render_scale: float = 0.5,
        num_retrieval: int = 3,
        max_candidates_to_test: int = 1,
        optimization_iterations: int = 1,
        min_correspondences: int = 20,
        min_inliers: int = 12,
        pnp_iterations: int = 2000,
        pnp_reprojection_error_px: float = 3.0,
        pnp_confidence: float = 0.9999,
        refine_pnp_lm: bool = True,
        min_depth: float = 1e-3,
        max_depth: float = 1e4,
        match_subsample: int = 8,
        reuse_last_pose: bool = False,
        use_vio: bool = False,
        vio_reset_after_failures: int = 3,
        vio_fallback_to_retrieval: bool = True,
        jpeg_quality: int = 95,
        save_debug: bool = False,
        debug_dir: str = "",
        logger: Optional[Callable] = None,
        warner: Optional[Callable] = None,
    ) -> None:
        self._log = logger or _noop
        self._warn = warner or self._log
        self.device = device
        self.camera_config = camera_config
        self.num_retrieval = max(1, int(num_retrieval))
        self.max_candidates_to_test = max(1, int(max_candidates_to_test))
        self.optimization_iterations = max(1, int(optimization_iterations))
        self.min_correspondences = max(6, int(min_correspondences))
        self.min_inliers = max(4, int(min_inliers))
        self.pnp_iterations = max(1, int(pnp_iterations))
        self.pnp_reprojection_error_px = float(pnp_reprojection_error_px)
        self.pnp_confidence = float(pnp_confidence)
        self.refine_pnp_lm = bool(refine_pnp_lm)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.match_subsample = int(match_subsample)
        self.render_use_principal_point = bool(render_use_principal_point)
        self.reuse_last_pose = bool(reuse_last_pose)
        self.use_vio = bool(use_vio)
        self.vio_fallback_to_retrieval = bool(vio_fallback_to_retrieval)
        self.vio_alignment = VioAlignment(reset_after_failures=vio_reset_after_failures)
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.save_debug = bool(save_debug)
        self.debug_dir = os.path.abspath(debug_dir) if debug_dir else ""
        self.last_pose_c2w: Optional[np.ndarray] = None
        self.colmap_model_dir: str = ""
        self.frame_counter = 0
        self.distortion = np.zeros((4, 1), dtype=np.float64)
        self._scaled_camera_cache: Dict[Tuple[int, int], CameraConfig] = {}

        self.runtime_dir = os.path.abspath(runtime_dir)
        os.makedirs(self.runtime_dir, exist_ok=True)
        if self.save_debug and self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

        self._log("Loading 3DGS map from %s", model_path)
        self.gaussian_map = GaussianMap(
            model_path=model_path,
            iteration=iteration,
            sh_degree=sh_degree,
            white_background=white_background,
            device=device,
            depth_mode=depth_mode,
            gs_repo_path=gs_repo_path,
            pipeline=PipelineConfig(),
            logger=self._log,
        )

        self.reference_poses, self.reference_camera, self.reference_source = (
            self._load_reference_poses(
                reference_source=reference_source,
                model_path=model_path,
                colmap_path=colmap_path,
                database_path=database_path,
                fallback_camera=camera_config.camera,
            )
        )
        self._log(
            "Loaded %d reference poses (source=%s)",
            len(self.reference_poses),
            self.reference_source,
        )

        self._log("Loading MASt3R model: %s", mast3r_model)
        from mast3r.model import AsymmetricMASt3R

        self.mast3r = AsymmetricMASt3R.from_pretrained(mast3r_model).to(device).eval()

        self.retrieval: Optional[NetVLADRetrieval] = None
        self.retrieval_source = "disabled"
        if enable_retrieval:
            descriptor_path = self._build_retrieval_database(
                retrieval_source=retrieval_source,
                model_path=model_path,
                database_path=database_path,
                reference_images_dir=reference_images_dir,
                retrieval_exclude=retrieval_exclude,
                retrieval_cache_dir=retrieval_cache_dir,
                rebuild_retrieval_db=rebuild_retrieval_db,
                db_render_scale=db_render_scale,
            )
            self.retrieval = NetVLADRetrieval(
                db_descriptor_path=descriptor_path,
                work_dir=self.runtime_dir,
                device=device,
                logger=self._log,
            )
        self._log("Localization engine ready")

    # -- startup ---------------------------------------------------------

    def _load_reference_poses(
        self,
        reference_source: str,
        model_path: str,
        colmap_path: str,
        database_path: str,
        fallback_camera: PinholeCamera,
    ) -> Tuple[Dict[str, np.ndarray], PinholeCamera, str]:
        """Resolve reference poses from whichever source the map provides.

        ``auto`` prefers a COLMAP sparse model: it is the reconstruction the map
        was trained from, so its world frame is the map frame, and it carries the
        principal point and distortion that ``cameras.json`` drops. A stock 3DGS
        model directory without a sparse model falls back to ``cameras.json``.
        """
        source = str(reference_source or "auto").strip().lower()
        if source not in ("auto", "colmap", "cameras_json", "loam"):
            raise ValueError(
                "reference_source must be 'auto', 'colmap', 'cameras_json' or "
                "'loam', got '{}'".format(reference_source)
            )

        if source == "loam":
            if not database_path:
                raise ValueError("reference_source='loam' requires database_path")
            poses, camera = load_reference_poses_from_loam(database_path)
            return poses, camera or fallback_camera, "loam"

        # Roots to search for a sparse model, most specific first.
        roots = [colmap_path, model_path, database_path]
        if source in ("auto", "colmap"):
            poses, camera, found = self._try_colmap_roots(roots)
            if poses is not None:
                self.colmap_model_dir = found
                self._log("Reference poses from the COLMAP model at %s", found)
                return poses, camera, "colmap"
            if source == "colmap":
                raise FileNotFoundError(
                    "reference_source='colmap' but no COLMAP model was found under "
                    "{}. Point ~colmap_path at the sparse model directory "
                    "(e.g. <dataset>/sparse/0).".format(
                        ", ".join(root for root in roots if root) or "<no path given>"
                    )
                )
            self._log("No COLMAP model found; falling back to cameras.json")

        poses, camera = load_reference_poses_from_cameras_json(model_path)
        return poses, camera or fallback_camera, "cameras_json"

    def _try_colmap_roots(self, roots: List[str]):
        for root in roots:
            if not root:
                continue
            try:
                return load_reference_poses_from_colmap(root)
            except FileNotFoundError:
                continue
        return None, None, None

    @property
    def colmap_image_roots(self) -> List[str]:
        """Dataset roots to look for raw reference images in.

        A model directory ``<dataset>/sparse/0`` normally sits beside
        ``<dataset>/images``, so the search climbs out of the sparse folder.
        """
        if not self.colmap_model_dir:
            return []
        first = os.path.abspath(self.colmap_model_dir)
        second = os.path.dirname(first)
        return [first, second, os.path.dirname(second)]

    def _build_retrieval_database(
        self,
        retrieval_source: str,
        model_path: str,
        database_path: str,
        reference_images_dir: str,
        retrieval_exclude: Optional[List[str]],
        retrieval_cache_dir: str,
        rebuild_retrieval_db: bool,
        db_render_scale: float,
    ) -> str:
        """Pick a retrieval database and return its descriptor file.

        ``auto`` order: a prepared ``<database_path>/sfm`` descriptor file, then
        the map's raw reference images, then rendering the reference poses. Raw
        images come before renders because the query is itself a photograph, so
        matching photographs against photographs avoids a render-to-real domain
        gap.
        """
        source = str(retrieval_source or "auto").strip().lower()
        if source not in ("auto", "folder", "images", "render"):
            raise ValueError(
                "retrieval_source must be 'auto', 'folder', 'images' or 'render', "
                "got '{}'".format(retrieval_source)
            )

        if source in ("auto", "folder"):
            descriptor_path = resolve_database_descriptors(database_path)
            if descriptor_path:
                self._log("Using prepared NetVLAD database: %s", descriptor_path)
                self.retrieval_source = "folder"
                return descriptor_path
            if source == "folder":
                raise FileNotFoundError(
                    "retrieval_source='folder' but no descriptors at "
                    "<database_path>/sfm/global-feats-netvlad.h5 (database_path="
                    "'{}')".format(database_path)
                )

        cache_dir = retrieval_cache_dir or os.path.join(
            os.path.abspath(model_path), "netvlad_cache"
        )

        if source in ("auto", "images"):
            image_dir = reference_images_dir or find_reference_image_dir(
                *([model_path, database_path] + self.colmap_image_roots)
            )
            if image_dir:
                self._log("Building the NetVLAD database from raw images: %s", image_dir)
                self.retrieval_source = "images"
                return build_database_from_images(
                    image_dir=image_dir,
                    reference_poses=self.reference_poses,
                    cache_dir=cache_dir,
                    force_rebuild=rebuild_retrieval_db,
                    exclude=retrieval_exclude,
                    logger=self._log,
                )
            if source == "images":
                raise FileNotFoundError(
                    "retrieval_source='images' but no reference image folder was "
                    "found; set ~reference_images_dir to the folder COLMAP was run on"
                )
            self._log("No raw reference images found; building the database from renders")

        self.retrieval_source = "render"
        return build_database_from_map(
            gaussian_map=self.gaussian_map,
            reference_poses=self.reference_poses,
            # The same centring rule as localization renders, so database views
            # and query-time views come from one convention.
            camera=self.render_camera_for(self.reference_camera),
            cache_dir=cache_dir,
            force_rebuild=rebuild_retrieval_db,
            render_scale=db_render_scale,
            logger=self._log,
        )

    # -- helpers ---------------------------------------------------------

    def camera_for(self, width: int, height: int) -> CameraConfig:
        """Intrinsics for an incoming image resolution, rescaled if it differs."""
        key = (int(width), int(height))
        if key not in self._scaled_camera_cache:
            scaled = self.camera_config.scaled_to(width, height)
            if scaled is not self.camera_config:
                self._warn(
                    "Incoming images are %dx%d but the camera config is %dx%d; "
                    "intrinsics were rescaled. Check this matches the real camera.",
                    width,
                    height,
                    self.camera_config.camera.width,
                    self.camera_config.camera.height,
                )
            self._scaled_camera_cache[key] = scaled
        return self._scaled_camera_cache[key]

    # -- pipeline stages -------------------------------------------------

    def build_correspondences(
        self,
        rendered_rgb,
        rendered_depth: np.ndarray,
        pose_c2w: np.ndarray,
        query_rgb: np.ndarray,
        render_camera: PinholeCamera,
        query_camera: PinholeCamera,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Match render against query, then lift matched render pixels to 3D.

        The two cameras are usually not the same model. Render pixels are
        backprojected through ``render_camera`` (whatever the map was rendered
        with), while the query pixels stay observations of ``query_camera`` and
        are handed to PnP under its true intrinsics.
        """
        render_rgb_uint8 = tensor_to_rgb_uint8(rendered_rgb)
        pixels_render, pixels_query = match_pair(
            self.mast3r,
            render_rgb_uint8,
            query_rgb,
            device=self.device,
            subsample=self.match_subsample,
        )
        empty = (np.zeros((0, 3), np.float32), np.zeros((0, 2), np.float32))
        if pixels_render.shape[0] == 0:
            return empty

        # Only matched pixels are backprojected; the original script built a
        # homogeneous point for every pixel of the render.
        render_u = np.rint(pixels_render[:, 0]).astype(np.int64)
        render_v = np.rint(pixels_render[:, 1]).astype(np.int64)
        depth_height, depth_width = rendered_depth.shape

        valid = (
            (render_u >= 0)
            & (render_u < depth_width)
            & (render_v >= 0)
            & (render_v < depth_height)
            & (pixels_query[:, 0] >= 0.0)
            & (pixels_query[:, 0] < query_camera.width)
            & (pixels_query[:, 1] >= 0.0)
            & (pixels_query[:, 1] < query_camera.height)
        )
        render_u, render_v = render_u[valid], render_v[valid]
        pixels_query = pixels_query[valid]
        if render_u.size == 0:
            return empty

        depth = rendered_depth[render_v, render_u].astype(np.float64)
        valid_depth = np.isfinite(depth) & (depth > self.min_depth) & (depth < self.max_depth)
        render_u, render_v = render_u[valid_depth], render_v[valid_depth]
        depth = depth[valid_depth]
        pixels_query = pixels_query[valid_depth]
        if depth.size == 0:
            return empty

        intrinsic = render_camera.K
        x = (render_u.astype(np.float64) - intrinsic[0, 2]) / intrinsic[0, 0] * depth
        y = (render_v.astype(np.float64) - intrinsic[1, 2]) / intrinsic[1, 1] * depth
        points_camera = np.column_stack((x, y, depth, np.ones_like(depth)))
        points_world = (np.asarray(pose_c2w, dtype=np.float64) @ points_camera.T).T[:, :3]

        return points_world.astype(np.float32), pixels_query.astype(np.float32)

    def run_pnp(
        self, points_world: np.ndarray, pixels_query: np.ndarray, intrinsic: np.ndarray
    ) -> LocalizationResult:
        num_matches = int(points_world.shape[0])
        if num_matches < self.min_correspondences:
            return LocalizationResult(
                success=False,
                num_matches=num_matches,
                error="Too few 2D-3D correspondences: {} < {}".format(
                    num_matches, self.min_correspondences
                ),
            )

        # EPnP is a global solver, so RANSAC needs no extrinsic guess; the pose is
        # refined afterwards with Levenberg-Marquardt over the inliers.
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=points_world,
            imagePoints=pixels_query,
            cameraMatrix=intrinsic,
            distCoeffs=self.distortion,
            iterationsCount=self.pnp_iterations,
            reprojectionError=self.pnp_reprojection_error_px,
            confidence=self.pnp_confidence,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None or len(inliers) < self.min_inliers:
            return LocalizationResult(
                success=False,
                num_matches=num_matches,
                num_inliers=0 if inliers is None else int(len(inliers)),
                error="PnP-RANSAC failed or gave too few inliers (need {})".format(
                    self.min_inliers
                ),
            )

        inlier_indices = inliers.reshape(-1)
        if self.refine_pnp_lm and len(inlier_indices) >= 6:
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    points_world[inlier_indices],
                    pixels_query[inlier_indices],
                    intrinsic,
                    self.distortion,
                    rvec,
                    tvec,
                )
            except cv2.error as exc:
                self._warn("solvePnPRefineLM failed: %s", exc)

        # solvePnP returns world-to-camera; the pipeline speaks camera-to-world.
        rotation_w2c, _ = cv2.Rodrigues(rvec)
        pose_c2w = np.eye(4, dtype=np.float64)
        pose_c2w[:3, :3] = rotation_w2c.T
        pose_c2w[:3, 3] = (-rotation_w2c.T @ tvec).reshape(3)

        projected, _ = cv2.projectPoints(
            points_world[inlier_indices], rvec, tvec, intrinsic, self.distortion
        )
        residuals = projected.reshape(-1, 2) - pixels_query[inlier_indices]
        rmse = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
        num_inliers = int(len(inlier_indices))

        return LocalizationResult(
            success=True,
            pose_c2w=pose_c2w,
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_ratio=num_inliers / float(max(1, num_matches)),
            reprojection_rmse_px=rmse,
        )

    def render_camera_for(self, query_camera: PinholeCamera) -> PinholeCamera:
        """The camera model to render the map with.

        Stock 3DGS never sees a principal point: ``dataset_readers`` turns the
        COLMAP intrinsics into FoVx/FoVy from the focal lengths alone, and its
        ``Camera`` class has no cx/cy. A map trained that way has absorbed any
        real principal-point offset into the gaussians, so it has to be rendered
        with a centred principal point or every render is shifted with respect to
        the map's own convention. On colmap_E2 that offset is 22.7 px.

        The query image is still a real observation of a real lens, so its true
        cx/cy stay in force for the PnP step.

        Set ``render_use_principal_point=True`` only for a map trained by a fork
        that genuinely models the principal point.
        """
        if self.render_use_principal_point:
            return query_camera
        return PinholeCamera(
            width=query_camera.width,
            height=query_camera.height,
            fx=query_camera.fx,
            fy=query_camera.fy,
            cx=query_camera.width / 2.0,
            cy=query_camera.height / 2.0,
        )

    def localize_from_initial_pose(
        self, initial_pose_c2w: np.ndarray, query_rgb: np.ndarray, camera: PinholeCamera
    ) -> LocalizationResult:
        """Render / match / PnP from a known starting pose, optionally iterated."""
        render_camera = self.render_camera_for(camera)
        current_pose = np.asarray(initial_pose_c2w, dtype=np.float64).copy()
        last_result = LocalizationResult(success=False, error="no iteration ran")
        timings = {"render_ms": 0.0, "match_ms": 0.0, "pnp_ms": 0.0}

        for _ in range(self.optimization_iterations):
            start = time.perf_counter()
            rendered_rgb, rendered_depth = self.gaussian_map.render(
                current_pose, render_camera
            )
            timings["render_ms"] += (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            points_world, pixels_query = self.build_correspondences(
                rendered_rgb, rendered_depth, current_pose, query_rgb, render_camera, camera
            )
            timings["match_ms"] += (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            last_result = self.run_pnp(points_world, pixels_query, camera.K)
            timings["pnp_ms"] += (time.perf_counter() - start) * 1000.0

            if not last_result.success or last_result.pose_c2w is None:
                last_result.timings_ms = timings
                return last_result
            current_pose = last_result.pose_c2w

        last_result.pose_c2w = current_pose
        last_result.timings_ms = timings
        return last_result

    # -- entry point -----------------------------------------------------

    def localize(
        self,
        image_bgr: np.ndarray,
        vio_pose_odom_cam: Optional[np.ndarray] = None,
        stamp: float = 0.0,
    ) -> LocalizationResult:
        """Relocalize one BGR frame against the map.

        ``vio_pose_odom_cam`` is the camera pose in the VIO frame at the moment
        this image was taken, already carrying any body-to-camera extrinsic. When
        it is given and the map-to-odom alignment is known, it primes the solve
        directly and no retrieval runs. Every successful fix re-derives the
        alignment, so the first frame (and any frame after the alignment is
        dropped) bootstraps it through retrieval.
        """
        start_time = time.perf_counter()
        self.frame_counter += 1
        try:
            if image_bgr is None or image_bgr.size == 0:
                raise ValueError("Received an empty image")
            if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
                raise ValueError("Expected a BGR image HxWx3, got {}".format(image_bgr.shape))

            height, width = image_bgr.shape[:2]
            config = self.camera_for(width, height)
            image_bgr = config.undistort(image_bgr)
            query_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            camera = config.camera

            use_vio = self.use_vio and vio_pose_odom_cam is not None
            best_failure: Optional[LocalizationResult] = None

            # Fast path 1: the VIO prior. Costs nothing but the prediction, and
            # replaces retrieval outright once the alignment is known.
            if use_vio:
                prediction = self.vio_alignment.predict(vio_pose_odom_cam)
                if prediction is not None:
                    result = self.localize_from_initial_pose(prediction, query_rgb, camera)
                    result.candidate_image = "vio_prior"
                    result.init_source = "vio"
                    if result.success:
                        result.vio_prediction_error_m = float(
                            np.linalg.norm(result.pose_c2w[:3, 3] - prediction[:3, 3])
                        )
                        return self._accept(
                            result, image_bgr, vio_pose_odom_cam, stamp, start_time
                        )
                    if self.vio_alignment.note_failure():
                        self._warn(
                            "Dropped the map-to-odom alignment after repeated failures "
                            "from the VIO prior; falling back to retrieval"
                        )
                    best_failure = result
                    if not self.vio_fallback_to_retrieval:
                        result.error = (
                            "VIO-primed localization failed and "
                            "vio_fallback_to_retrieval is off: {}".format(result.error)
                        )
                        result.processing_time_ms = (
                            time.perf_counter() - start_time
                        ) * 1000.0
                        return result

            # Fast path 2: the previous pose, for a device without VIO.
            if self.reuse_last_pose and self.last_pose_c2w is not None:
                result = self.localize_from_initial_pose(
                    self.last_pose_c2w.copy(), query_rgb, camera
                )
                result.candidate_image = "previous_pose"
                result.init_source = "previous_pose"
                if result.success:
                    return self._accept(result, image_bgr, vio_pose_odom_cam, stamp, start_time)
                if best_failure is None or result.num_inliers > best_failure.num_inliers:
                    best_failure = result

            if self.retrieval is None:
                raise RuntimeError(
                    "Retrieval is disabled and no usable initial pose was available"
                )

            start = time.perf_counter()
            candidates = self.retrieval.retrieve(image_bgr, self.num_retrieval)
            retrieval_ms = (time.perf_counter() - start) * 1000.0

            for candidate in candidates[: self.max_candidates_to_test]:
                key = resolve_reference_name(candidate, self.reference_poses)
                result = self.localize_from_initial_pose(
                    self.reference_poses[key], query_rgb, camera
                )
                result.candidate_image = key
                result.init_source = "retrieval"
                result.timings_ms["retrieval_ms"] = retrieval_ms
                if result.success:
                    return self._accept(result, image_bgr, vio_pose_odom_cam, stamp, start_time)
                if best_failure is None or result.num_inliers > best_failure.num_inliers:
                    best_failure = result

            result = best_failure or LocalizationResult(
                success=False, error="No localization initialization was available"
            )
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000.0
            return result
        except Exception as exc:  # keep a long-running server alive
            self._warn("localize() failed: %s", traceback.format_exc())
            return LocalizationResult(
                success=False,
                processing_time_ms=(time.perf_counter() - start_time) * 1000.0,
                error="{}: {}".format(type(exc).__name__, exc),
            )

    def _accept(
        self,
        result: LocalizationResult,
        image_bgr: np.ndarray,
        vio_pose_odom_cam: Optional[np.ndarray],
        stamp: float,
        start_time: float,
    ) -> LocalizationResult:
        """Book-keeping common to every successful path."""
        self.last_pose_c2w = result.pose_c2w.copy()
        if self.use_vio and vio_pose_odom_cam is not None:
            # Re-derive the alignment from this fix, whichever path produced it.
            # Always from the newest pair, never integrated, so VIO drift is
            # corrected rather than accumulated.
            self.vio_alignment.update(result.pose_c2w, vio_pose_odom_cam, stamp)
        result.processing_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.save_debug_outputs(image_bgr, result)
        return result

    def save_debug_outputs(self, query_bgr: np.ndarray, result: LocalizationResult) -> None:
        if not self.save_debug or not self.debug_dir or result.pose_c2w is None:
            return
        stem = os.path.join(self.debug_dir, "{:08d}".format(self.frame_counter))
        cv2.imwrite(stem + "_query.jpg", query_bgr)
        np.savetxt(stem + "_pose_c2w.txt", result.pose_c2w, fmt="%.9f")
        try:
            config = self.camera_for(query_bgr.shape[1], query_bgr.shape[0])
            rendered_rgb, rendered_depth = self.gaussian_map.render(
                result.pose_c2w, self.render_camera_for(config.camera)
            )
            rgb = tensor_to_rgb_uint8(rendered_rgb)
            cv2.imwrite(stem + "_render.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            np.save(stem + "_render_depth.npy", rendered_depth)
        except Exception as exc:
            self._warn("Could not save debug render: %s", exc)


def pose_error(estimated_c2w: np.ndarray, reference_c2w: np.ndarray) -> Tuple[float, float]:
    """Translation (metres) and rotation (degrees) between two c2w poses."""
    translation = float(np.linalg.norm(estimated_c2w[:3, 3] - reference_c2w[:3, 3]))
    relative = estimated_c2w[:3, :3].T @ reference_c2w[:3, :3]
    cosine = (np.trace(relative) - 1.0) / 2.0
    rotation = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return translation, rotation
