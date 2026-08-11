#!/usr/bin/env python3
"""Offline geometry self-test for the relocalization pipeline. No ROS needed.

Renders a reference view B, hands it to the pipeline as the query, and starts the
solve from a *different* pose A. The pipeline then has to recover B. Because the
query and the map are the same 3DGS model, anything left in the pose error comes
from the geometry chain itself: the depth convention, the principal point, the
render pose convention, and the MASt3R pixel back-mapping.

Starting from a different pose is what makes this test meaningful. Initializing
at B is degenerate: matches are the identity, so every depth scale reprojects to
the same pixels and B is recovered no matter what the depth means.

Run it after changing anything geometric:

    python selftest_localization.py --model_path data/colmap_E2

Use ``--depth_mode both`` to compare the two rasterizer depth conventions.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_config import CameraConfig  # noqa: E402
from gs_render_backend import PinholeCamera  # noqa: E402
from localization_pipeline import LocalizationEngine, pose_error  # noqa: E402
from mast3r_matching import tensor_to_rgb_uint8  # noqa: E402


def log(fmt, *args):
    print(fmt % args if args else fmt, flush=True)


def reference_camera_for(model_path: str) -> PinholeCamera:
    """Intrinsics of the map's reference camera, from COLMAP or cameras.json."""
    from gs_render_backend import (
        load_reference_poses_from_cameras_json,
        load_reference_poses_from_colmap,
    )

    try:
        _, camera, _ = load_reference_poses_from_colmap(model_path)
        return camera
    except FileNotFoundError:
        _, camera = load_reference_poses_from_cameras_json(model_path)
        return camera


def build_engine(args, depth_mode: str) -> LocalizationEngine:
    reference = reference_camera_for(args.model_path)
    width = args.width or reference.width
    height = args.height or reference.height
    # The synthetic query is a render, so it is an observation of whatever camera
    # the pipeline renders with. That camera is centred (stock 3DGS has no
    # principal point), so the query camera is centred here too — otherwise a
    # COLMAP cx/cy offset would show up as a constant pose bias and mask the
    # geometry error this test is meant to measure. Real queries keep their true
    # cx/cy; see LocalizationEngine.render_camera_for.
    camera = PinholeCamera(
        width=int(width),
        height=int(height),
        fx=reference.fx * (float(width) / float(reference.width)),
        fy=reference.fy * (float(height) / float(reference.height)),
        cx=float(width) / 2.0,
        cy=float(height) / 2.0,
    )

    return LocalizationEngine(
        model_path=args.model_path,
        camera_config=CameraConfig(camera=camera),
        runtime_dir=args.runtime_dir,
        iteration=args.iteration,
        device=args.device,
        depth_mode=depth_mode,
        enable_retrieval=False,  # poses come from the reference set directly
        optimization_iterations=args.optimization_iterations,
        min_correspondences=args.min_correspondences,
        min_inliers=args.min_inliers,
        pnp_reprojection_error_px=args.pnp_reprojection_error_px,
        logger=log if args.verbose else None,
        warner=log,
    )


def neighbour_by_distance(
    poses: Dict[str, np.ndarray], names: List[str], index: int, step: int
) -> str:
    """The ``step``-th nearest reference camera to ``names[index]``.

    Neighbours are chosen by camera centre rather than by position in the sorted
    name list: filenames like ``frame0, frame1, frame10, frame100`` sort
    lexicographically, so "two names later" can be metres away, which turns this
    into a test of retrieval range instead of a test of the geometry chain.
    """
    centres = np.array([poses[name][:3, 3] for name in names])
    distances = np.linalg.norm(centres - centres[index], axis=1)
    order = np.argsort(distances)  # order[0] is the query itself
    return names[int(order[min(step, len(order) - 1)])]


def run(engine: LocalizationEngine, args, label: str) -> None:
    names = sorted(engine.reference_poses.keys())
    if len(names) < args.step + 2:
        raise RuntimeError("Not enough reference cameras for this test")

    camera = engine.camera_config.camera
    indices = np.linspace(0, len(names) - 1, args.num_pairs).astype(int)

    translations, rotations, inliers, rmses = [], [], [], []
    log("")
    log("=== depth_mode=%s ===", label)
    log("%-28s %10s %10s %9s %9s %8s", "query", "trans[m]", "rot[deg]", "inliers", "matches", "rmse")

    initial_names = {
        int(index): neighbour_by_distance(
            engine.reference_poses, names, int(index), args.step
        )
        for index in indices
    }

    for index in indices:
        name_query = names[int(index)]
        name_init = initial_names[int(index)]
        pose_query = engine.reference_poses[name_query]
        pose_init = engine.reference_poses[name_init]

        rendered_rgb, _ = engine.gaussian_map.render(pose_query, camera)
        query_rgb = tensor_to_rgb_uint8(rendered_rgb)

        result = engine.localize_from_initial_pose(pose_init, query_rgb, camera)
        if not result.success:
            log("%-28s %10s %10s %9s %9d %8s  <- %s",
                name_query, "FAIL", "-", "-", result.num_matches, "-", result.error)
            continue

        translation, rotation = pose_error(result.pose_c2w, pose_query)
        translations.append(translation)
        rotations.append(rotation)
        inliers.append(result.num_inliers)
        rmses.append(result.reprojection_rmse_px)
        log("%-28s %10.4f %10.4f %9d %9d %8.3f",
            name_query, translation, rotation, result.num_inliers,
            result.num_matches, result.reprojection_rmse_px)

    if not translations:
        log("all pairs failed for depth_mode=%s", label)
        return

    baseline = np.mean([
        np.linalg.norm(
            engine.reference_poses[names[int(i)]][:3, 3]
            - engine.reference_poses[initial_names[int(i)]][:3, 3]
        )
        for i in indices
    ])
    log("-" * 82)
    log("depth_mode=%-8s solved %d/%d   median trans=%.4f m  median rot=%.4f deg",
        label, len(translations), len(indices), np.median(translations), np.median(rotations))
    log("               mean initial-pose offset=%.3f m   median inliers=%d   median rmse=%.3f px",
        baseline, int(np.median(inliers)), np.median(rmses))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", "-m", default="data/colmap_E2")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--depth_mode",
        default="inverse",
        choices=["inverse", "linear", "both"],
        help="'both' runs the test twice to show which convention the rasterizer uses",
    )
    parser.add_argument("--num_pairs", type=int, default=6)
    parser.add_argument(
        "--step",
        type=int,
        default=2,
        help="initial pose = the step-th nearest reference camera to the query",
    )
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--optimization_iterations", type=int, default=1)
    parser.add_argument("--min_correspondences", type=int, default=20)
    parser.add_argument("--min_inliers", type=int, default=12)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=3.0)
    parser.add_argument("--runtime_dir", default="/tmp/3dgs_localization_selftest")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    modes = ["inverse", "linear"] if args.depth_mode == "both" else [args.depth_mode]
    for mode in modes:
        engine = build_engine(args, mode)
        run(engine, args, mode)


if __name__ == "__main__":
    main()
