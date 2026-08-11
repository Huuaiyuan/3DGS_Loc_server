#!/usr/bin/env python3
"""COLMAP sparse-model reader (``cameras`` / ``images``), text and binary.

A COLMAP reconstruction is the canonical way to carry reference poses: it is
what 3DGS was trained from in the first place, it is written by a tool outside
this repository, and it keeps the full pinhole intrinsics including the
principal point and the distortion model. ``cameras.json`` in a trained 3DGS
model directory is a lossy re-export of exactly this information (no principal
point, no distortion) and only exists if the training run wrote one.

Both encodings of the same model are supported. ``.txt`` is preferred when both
are present because it is readable and diffable; ``.bin`` is read when only that
exists, so a model straight out of COLMAP works without a conversion step. To
produce the text form yourself:

    colmap model_converter --input_path sparse/0 --output_path sparse/0 \
        --output_type TXT

Pose convention
---------------
COLMAP stores each image as the *world-to-camera* rotation quaternion ``QVEC``
(w, x, y, z) and translation ``TVEC``: a world point maps to the camera frame as
``x_cam = R(QVEC) @ x_world + TVEC``. :func:`ColmapImage.pose_c2w` inverts that,
so everything leaving this module is camera-to-world, matching the rest of the
pipeline.

This module deliberately imports nothing from this repository, so it stays a
leaf that both ``camera_config`` and ``gs_render_backend`` can build on.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# model_id -> (name, number of parameters); the order COLMAP defines in
# src/colmap/sensor/models.h.
CAMERA_MODEL_IDS: Dict[int, Tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

CAMERA_MODEL_NAMES: Dict[str, int] = {
    name: model_id for model_id, (name, _) in CAMERA_MODEL_IDS.items()
}

# Distortion model names as camera_config understands them.
_NONE = "none"
_RADTAN = "plumb_bob"
_FISHEYE = "equidistant"


@dataclass
class ColmapCamera:
    """One COLMAP camera: a projection model and its parameters."""

    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def pinhole(self) -> Tuple[float, float, float, float]:
        """(fx, fy, cx, cy) for this model.

        Every COLMAP model starts with either ``f, cx, cy`` or ``fx, fy, cx, cy``
        and puts its distortion coefficients after that, so the pinhole part can
        be pulled out uniformly.
        """
        params = [float(value) for value in self.params]
        if self.model not in CAMERA_MODEL_NAMES:
            raise ValueError("Unknown COLMAP camera model: {}".format(self.model))
        if self.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
            return params[0], params[1], params[2], params[3]
        # Single-focal models: f, cx, cy, ...
        return params[0], params[0], params[1], params[2]

    def distortion(self) -> Tuple[str, List[float]]:
        """(distortion_model, coefficients) in OpenCV order.

        OpenCV's radial-tangential order is [k1, k2, p1, p2, k3, k4, k5, k6],
        which is *not* COLMAP's order for FULL_OPENCV, so the coefficients are
        reordered rather than passed through.
        """
        params = [float(value) for value in self.params]
        model = self.model
        if model in ("SIMPLE_PINHOLE", "PINHOLE"):
            return _NONE, []
        if model == "SIMPLE_RADIAL":  # f, cx, cy, k
            return _RADTAN, [params[3], 0.0, 0.0, 0.0]
        if model == "RADIAL":  # f, cx, cy, k1, k2
            return _RADTAN, [params[3], params[4], 0.0, 0.0]
        if model == "OPENCV":  # fx, fy, cx, cy, k1, k2, p1, p2
            return _RADTAN, params[4:8]
        if model == "FULL_OPENCV":  # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
            k1, k2, p1, p2, k3, k4, k5, k6 = params[4:12]
            return _RADTAN, [k1, k2, p1, p2, k3, k4, k5, k6]
        if model == "OPENCV_FISHEYE":  # fx, fy, cx, cy, k1..k4
            return _FISHEYE, params[4:8]
        if model == "SIMPLE_RADIAL_FISHEYE":  # f, cx, cy, k
            return _FISHEYE, [params[3], 0.0, 0.0, 0.0]
        if model == "RADIAL_FISHEYE":  # f, cx, cy, k1, k2
            return _FISHEYE, [params[3], params[4], 0.0, 0.0]
        raise ValueError(
            "COLMAP camera model '{}' has no OpenCV distortion equivalent; "
            "undistort the images first or re-run COLMAP with an OPENCV model".format(model)
        )


@dataclass
class ColmapImage:
    """One registered image: its name and its world-to-camera pose."""

    id: int
    qvec: np.ndarray  # (w, x, y, z), world-to-camera
    tvec: np.ndarray  # world-to-camera translation
    camera_id: int
    name: str

    def rotation_w2c(self) -> np.ndarray:
        return qvec_to_rotation(self.qvec)

    def pose_w2c(self) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self.rotation_w2c()
        pose[:3, 3] = np.asarray(self.tvec, dtype=np.float64).reshape(3)
        return pose

    def pose_c2w(self) -> np.ndarray:
        rotation = self.rotation_w2c()
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation.T
        pose[:3, 3] = -rotation.T @ np.asarray(self.tvec, dtype=np.float64).reshape(3)
        return pose


def qvec_to_rotation(qvec: Sequence[float]) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) to a rotation matrix."""
    w, x, y, z = [float(value) for value in qvec]
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise ValueError("Zero-norm quaternion in the COLMAP model")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


# -- text format ---------------------------------------------------------


def _text_lines(path: str):
    with open(path, "r") as handle:
        for raw in handle:
            line = raw.strip()
            if line and not line.startswith("#"):
                yield line


def read_cameras_text(path: str) -> Dict[int, ColmapCamera]:
    cameras: Dict[int, ColmapCamera] = {}
    for line in _text_lines(path):
        fields = line.split()
        camera_id = int(fields[0])
        cameras[camera_id] = ColmapCamera(
            id=camera_id,
            model=fields[1].upper(),
            width=int(fields[2]),
            height=int(fields[3]),
            params=np.array([float(value) for value in fields[4:]], dtype=np.float64),
        )
    return cameras


def read_images_text(path: str) -> Dict[int, ColmapImage]:
    """Read ``images.txt``, skipping the POINTS2D line that follows each image."""
    images: Dict[int, ColmapImage] = {}
    lines = list(_text_lines(path))
    # Two lines per image, but a model exported without 2D observations can have
    # an empty (therefore skipped) second line, so entries are detected by shape
    # rather than by position: a pose line has >= 10 fields and a numeric id.
    for line in lines:
        fields = line.split()
        if len(fields) < 10:
            continue
        try:
            image_id = int(fields[0])
            qvec = np.array([float(value) for value in fields[1:5]], dtype=np.float64)
            tvec = np.array([float(value) for value in fields[5:8]], dtype=np.float64)
            camera_id = int(fields[8])
        except ValueError:
            continue  # a POINTS2D line
        # Names may contain spaces, so take everything after the 9th field.
        name = line.split(None, 9)[9].strip()
        images[image_id] = ColmapImage(
            id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=name
        )
    return images


# -- binary format -------------------------------------------------------


def _read_struct(handle, fmt: str):
    size = struct.calcsize(fmt)
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("Truncated COLMAP binary file")
    return struct.unpack(fmt, data)


def read_cameras_binary(path: str) -> Dict[int, ColmapCamera]:
    cameras: Dict[int, ColmapCamera] = {}
    with open(path, "rb") as handle:
        (count,) = _read_struct(handle, "<Q")
        for _ in range(count):
            camera_id, model_id, width, height = _read_struct(handle, "<iiQQ")
            if model_id not in CAMERA_MODEL_IDS:
                raise ValueError("Unknown COLMAP camera model id {}".format(model_id))
            model, num_params = CAMERA_MODEL_IDS[model_id]
            params = _read_struct(handle, "<" + "d" * num_params)
            cameras[camera_id] = ColmapCamera(
                id=camera_id,
                model=model,
                width=int(width),
                height=int(height),
                params=np.array(params, dtype=np.float64),
            )
    return cameras


def read_images_binary(path: str) -> Dict[int, ColmapImage]:
    images: Dict[int, ColmapImage] = {}
    with open(path, "rb") as handle:
        (count,) = _read_struct(handle, "<Q")
        for _ in range(count):
            fields = _read_struct(handle, "<idddddddi")
            image_id = int(fields[0])
            qvec = np.array(fields[1:5], dtype=np.float64)
            tvec = np.array(fields[5:8], dtype=np.float64)
            camera_id = int(fields[8])
            name_bytes = bytearray()
            while True:
                char = handle.read(1)
                if not char or char == b"\x00":
                    break
                name_bytes += char
            (num_points2d,) = _read_struct(handle, "<Q")
            handle.seek(24 * num_points2d, os.SEEK_CUR)  # (x, y, point3D_id) each
            images[image_id] = ColmapImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name_bytes.decode("utf-8"),
            )
    return images


# -- model directory -----------------------------------------------------


@dataclass
class ColmapModel:
    path: str
    format: str  # "txt" or "bin"
    cameras: Dict[int, ColmapCamera]
    images: Dict[int, ColmapImage]

    def poses_c2w(self) -> Dict[str, np.ndarray]:
        """Camera-to-world poses keyed by image name."""
        return {image.name: image.pose_c2w() for image in self.images.values()}

    def camera_for_image(self, name: str) -> ColmapCamera:
        for image in self.images.values():
            if image.name == name:
                return self.cameras[image.camera_id]
        raise KeyError("No image named '{}' in {}".format(name, self.path))

    def dominant_camera(self) -> ColmapCamera:
        """The camera most images were taken with.

        Reference renders need one intrinsic set; picking the most common camera
        is right for the usual single-camera capture and a sane default when a
        few frames came from a second sensor.
        """
        if not self.cameras:
            raise RuntimeError("COLMAP model has no cameras: {}".format(self.path))
        counts: Dict[int, int] = {}
        for image in self.images.values():
            counts[image.camera_id] = counts.get(image.camera_id, 0) + 1
        if not counts:
            return next(iter(self.cameras.values()))
        best = max(counts.items(), key=lambda item: item[1])[0]
        return self.cameras[best]

    def describe(self) -> str:
        camera = self.dominant_camera()
        fx, fy, cx, cy = camera.pinhole()
        return (
            "{} images, {} camera(s), model={} {}x{} fx={:.3f} fy={:.3f} "
            "cx={:.3f} cy={:.3f} [{}]".format(
                len(self.images),
                len(self.cameras),
                camera.model,
                camera.width,
                camera.height,
                fx,
                fy,
                cx,
                cy,
                self.format,
            )
        )


def detect_model_format(path: str) -> Optional[str]:
    """Return ``"txt"``, ``"bin"`` or None for a candidate model directory."""
    if not os.path.isdir(path):
        return None
    has_text = os.path.isfile(os.path.join(path, "cameras.txt")) and os.path.isfile(
        os.path.join(path, "images.txt")
    )
    if has_text:
        return "txt"
    has_binary = os.path.isfile(os.path.join(path, "cameras.bin")) and os.path.isfile(
        os.path.join(path, "images.bin")
    )
    return "bin" if has_binary else None


def read_model(path: str) -> ColmapModel:
    """Read a COLMAP sparse model directory, preferring the text encoding."""
    path = os.path.abspath(path)
    model_format = detect_model_format(path)
    if model_format is None:
        raise FileNotFoundError(
            "No COLMAP model in {} (expected cameras.txt/images.txt or "
            "cameras.bin/images.bin)".format(path)
        )
    if model_format == "txt":
        cameras = read_cameras_text(os.path.join(path, "cameras.txt"))
        images = read_images_text(os.path.join(path, "images.txt"))
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras.bin"))
        images = read_images_binary(os.path.join(path, "images.bin"))
    if not images:
        raise RuntimeError("COLMAP model has no registered images: {}".format(path))
    if not cameras:
        raise RuntimeError("COLMAP model has no cameras: {}".format(path))
    missing = {image.camera_id for image in images.values()} - set(cameras)
    if missing:
        raise RuntimeError(
            "COLMAP model {} references camera ids {} that cameras.{} does not "
            "define".format(path, sorted(missing), model_format)
        )
    return ColmapModel(path=path, format=model_format, cameras=cameras, images=images)


# Where a sparse model usually sits relative to a 3DGS model directory or a
# dataset root. ``sparse/0`` is what COLMAP's automatic reconstruction writes and
# what 3DGS' ``readColmapSceneInfo`` expects.
MODEL_SUBDIRS = (
    "",
    "sparse/0",
    "sparse",
    "colmap/sparse/0",
    "colmap/sparse",
    "sfm/sparse/0",
    "sparse/txt",
)


def find_model_dir(*roots: str) -> Optional[str]:
    """First readable COLMAP model under any of ``roots``.

    Each root is tried against :data:`MODEL_SUBDIRS`, in order, so passing a
    model directory, a dataset root, or the dataset's ``sparse/`` parent all
    work.
    """
    for root in roots:
        if not root:
            continue
        root = os.path.abspath(root)
        for subdir in MODEL_SUBDIRS:
            candidate = os.path.join(root, subdir) if subdir else root
            if detect_model_format(candidate) is not None:
                return candidate
    return None


def load_reference_poses(path: str) -> Tuple[Dict[str, np.ndarray], ColmapCamera, ColmapModel]:
    """Camera-to-world poses by image name, plus the model's dominant camera."""
    model = read_model(path)
    return model.poses_c2w(), model.dominant_camera(), model


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a COLMAP sparse model")
    parser.add_argument("path", help="model directory, or a dataset root to search")
    args = parser.parse_args()

    found = find_model_dir(args.path)
    if found is None:
        raise SystemExit("No COLMAP model found under {}".format(args.path))
    model = read_model(found)
    print("{}: {}".format(found, model.describe()))
    camera = model.dominant_camera()
    print("distortion: {} {}".format(*camera.distortion()))
    for image in sorted(model.images.values(), key=lambda item: item.name)[:5]:
        centre = image.pose_c2w()[:3, 3]
        print("  {:<24s} centre [{: .3f} {: .3f} {: .3f}]".format(image.name, *centre))
    print("  ... {} images total".format(len(model.images)))


if __name__ == "__main__":
    main()
