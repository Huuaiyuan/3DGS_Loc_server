#!/usr/bin/env python3
"""Query-camera intrinsics: loading, rescaling and undistortion.

Accepts the LOAM/r3live key names the rest of this repository uses
(``cam_fx`` ...) as well as plain ``fx``/``width`` names, so a calibration file
written by hand or exported from a ROS ``CameraInfo`` both work.

The 3DGS map is rendered with a pinhole model, so a query image from a lens with
real distortion has to be undistorted before matching. Undistortion is done with
the *same* camera matrix, which keeps the intrinsics valid afterwards.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

from gs_render_backend import PinholeCamera

if TYPE_CHECKING:  # avoids a hard import cycle at runtime
    from colmap_model import ColmapCamera

_ALIASES = {
    "width": ("cam_width", "width", "image_width"),
    "height": ("cam_height", "height", "image_height"),
    "fx": ("cam_fx", "fx"),
    "fy": ("cam_fy", "fy"),
    "cx": ("cam_cx", "cx"),
    "cy": ("cam_cy", "cy"),
}

NO_DISTORTION = ("none", "", "pinhole", "no_distortion")
RADTAN = ("plumb_bob", "radtan", "rational_polynomial", "radial_tangential")
FISHEYE = ("equidistant", "fisheye", "kannala_brandt")


class CameraConfig:
    """Intrinsics plus an optional distortion model for one physical camera."""

    def __init__(
        self,
        camera: PinholeCamera,
        distortion_model: str = "none",
        distortion_coeffs: Optional[np.ndarray] = None,
    ) -> None:
        self.camera = camera
        self.distortion_model = str(distortion_model or "none").strip().lower()
        coeffs = np.asarray(
            distortion_coeffs if distortion_coeffs is not None else [], dtype=np.float64
        ).reshape(-1)
        self.distortion_coeffs = coeffs

        if self.distortion_model not in NO_DISTORTION + RADTAN + FISHEYE:
            raise ValueError(
                "Unsupported distortion_model '{}'. Use one of: {}".format(
                    self.distortion_model, ", ".join(("none",) + RADTAN[:2] + FISHEYE[:2])
                )
            )
        if self.distortion_model in FISHEYE and coeffs.size not in (0, 4):
            raise ValueError("Fisheye distortion needs exactly 4 coefficients")

        self._map_x = None
        self._map_y = None
        self._map_shape = None

    @property
    def has_distortion(self) -> bool:
        return (
            self.distortion_model not in NO_DISTORTION
            and self.distortion_coeffs.size > 0
            and bool(np.any(np.abs(self.distortion_coeffs) > 1e-12))
        )

    def scaled_to(self, width: int, height: int) -> "CameraConfig":
        """Rescale to a different image resolution.

        Distortion coefficients are dimensionless in normalized image
        coordinates, so they carry over unchanged.
        """
        if int(width) == int(self.camera.width) and int(height) == int(self.camera.height):
            return self
        return CameraConfig(
            camera=self.camera.scaled_to(width, height),
            distortion_model=self.distortion_model,
            distortion_coeffs=self.distortion_coeffs,
        )

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Undistort an image, keeping the same camera matrix.

        The remap tables are built once and reused, since every frame from a
        given camera shares them.
        """
        if not self.has_distortion:
            return image
        height, width = image.shape[:2]
        if self._map_shape != (height, width):
            camera = self.camera.scaled_to(width, height)
            K = camera.K
            if self.distortion_model in FISHEYE:
                self._map_x, self._map_y = cv2.fisheye.initUndistortRectifyMap(
                    K, self.distortion_coeffs.reshape(4, 1), np.eye(3), K, (width, height), cv2.CV_32FC1
                )
            else:
                self._map_x, self._map_y = cv2.initUndistortRectifyMap(
                    K, self.distortion_coeffs, np.eye(3), K, (width, height), cv2.CV_32FC1
                )
            self._map_shape = (height, width)
        return cv2.remap(image, self._map_x, self._map_y, interpolation=cv2.INTER_LINEAR)

    def describe(self) -> str:
        camera = self.camera
        text = "{}x{} fx={:.3f} fy={:.3f} cx={:.3f} cy={:.3f}".format(
            camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy
        )
        if self.has_distortion:
            text += " distortion={}{}".format(
                self.distortion_model, np.array2string(self.distortion_coeffs, precision=5)
            )
        return text


def _pick(data: Dict[str, Any], field: str) -> Any:
    for key in _ALIASES[field]:
        if key in data:
            return data[key]
    return None


def camera_config_from_dict(data: Dict[str, Any], source: str = "<dict>") -> CameraConfig:
    values = {}
    for field in ("width", "height", "fx", "fy"):
        value = _pick(data, field)
        if value is None:
            raise KeyError("Camera config {} is missing '{}'".format(source, field))
        values[field] = float(value)

    # A missing principal point defaults to the image centre.
    cx = _pick(data, "cx")
    cy = _pick(data, "cy")
    values["cx"] = float(cx) if cx is not None else values["width"] / 2.0
    values["cy"] = float(cy) if cy is not None else values["height"] / 2.0

    camera = PinholeCamera(
        width=int(round(values["width"])),
        height=int(round(values["height"])),
        fx=values["fx"],
        fy=values["fy"],
        cx=values["cx"],
        cy=values["cy"],
    )
    coeffs = data.get("distortion_coeffs", data.get("distortion", []))
    model = data.get("distortion_model", "none")
    if coeffs is None:
        coeffs = []
    return CameraConfig(camera=camera, distortion_model=model, distortion_coeffs=coeffs)


def load_camera_config(path: str) -> CameraConfig:
    """Load a camera calibration YAML."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError("Camera config not found: {}".format(path))
    with open(path, "r") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Camera config must contain a mapping: {}".format(path))
    # Tolerate a nested block, e.g. {camera: {...}}.
    if "camera" in data and isinstance(data["camera"], dict):
        data = dict(data["camera"])
    return camera_config_from_dict(data, source=path)


def camera_config_from_colmap(camera: "ColmapCamera") -> CameraConfig:
    """Convert a COLMAP camera into a :class:`CameraConfig`."""
    fx, fy, cx, cy = camera.pinhole()
    distortion_model, coeffs = camera.distortion()
    return CameraConfig(
        camera=PinholeCamera(
            width=int(camera.width), height=int(camera.height), fx=fx, fy=fy, cx=cx, cy=cy
        ),
        distortion_model=distortion_model,
        distortion_coeffs=coeffs,
    )


def load_colmap_camera(path: str) -> CameraConfig:
    """Read intrinsics from a COLMAP model.

    ``path`` may be a ``cameras.txt`` file, a model directory holding
    ``cameras.txt``/``cameras.bin``, or a dataset root to search for one (see
    :func:`colmap_model.find_model_dir`). When the model has several cameras the
    one used by the most images wins.
    """
    from colmap_model import find_model_dir, read_cameras_text, read_model

    path = os.path.abspath(path)
    if os.path.isfile(path):
        cameras = read_cameras_text(path)
        if not cameras:
            raise ValueError("No camera line found in {}".format(path))
        camera = cameras[sorted(cameras)[0]]
    else:
        found = find_model_dir(path)
        if found is None:
            raise FileNotFoundError("No COLMAP model found under {}".format(path))
        camera = read_model(found).dominant_camera()
    return camera_config_from_colmap(camera)


# Kept for callers that pass a cameras.txt explicitly.
load_colmap_camera_txt = load_colmap_camera


def save_camera_config(config: CameraConfig, path: str, comment: str = "") -> None:
    """Write a camera config YAML in the form ``load_camera_config`` expects."""
    camera = config.camera
    lines = []
    if comment:
        lines.extend("# {}".format(part) for part in comment.splitlines())
    lines.extend(
        [
            "cam_width: {}".format(camera.width),
            "cam_height: {}".format(camera.height),
            "cam_fx: {:.6f}".format(camera.fx),
            "cam_fy: {:.6f}".format(camera.fy),
            "cam_cx: {:.6f}".format(camera.cx),
            "cam_cy: {:.6f}".format(camera.cy),
            "distortion_model: {}".format(config.distortion_model),
            "distortion_coeffs: [{}]".format(
                ", ".join("{:.8f}".format(value) for value in config.distortion_coeffs)
            ),
            "",
        ]
    )
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
