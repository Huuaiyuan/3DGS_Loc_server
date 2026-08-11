#!/usr/bin/env python3
"""Using a VIO stream as the initial-pose source instead of image retrieval.

A device that sends query images usually also runs VIO (a drone, a handheld
rig). VIO is locally accurate but drifts and starts at an arbitrary origin, so
its poses live in their own frame — call it *odom*. The map lives in the COLMAP
world frame. The two are related by one rigid transform:

    T_map_cam(t) = T_map_odom @ T_odom_cam(t)

``T_map_odom`` is unknown until the first successful relocalization, which does
use NetVLAD retrieval. After that single pair of poses,

    T_map_odom = T_map_cam(k) @ T_odom_cam(k)^-1

and every later frame can be primed by predicting ``T_map_odom @ T_odom_cam(t)``
instead of running retrieval. Each new successful fix recomputes the transform,
which is what keeps VIO drift from accumulating: the alignment is always derived
from the most recent fix rather than integrated over time.

Two things this does *not* assume:

* that VIO is metric-consistent with the map over long distances. Only the
  relative motion since the last fix has to be good, and that is VIO's strength.
* that VIO reports the camera optical frame. It normally reports a body frame,
  so the caller applies its own body-to-camera extrinsic before handing a pose
  in here. A wrong extrinsic does not bias the result globally (it is absorbed
  into ``T_map_odom``) but it does corrupt the *relative* prediction, in
  proportion to how much the body rotated since the last fix.

Nothing here imports ROS, so the alignment can be exercised offline.
"""

from __future__ import annotations

import bisect
import threading
from typing import List, Optional, Tuple

import numpy as np


def invert_pose(pose: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform without a general matrix inverse."""
    pose = np.asarray(pose, dtype=np.float64)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def pose_from_position_quaternion(position, quaternion) -> np.ndarray:
    """4x4 pose from a translation and an (x, y, z, w) quaternion."""
    from scipy.spatial.transform import Rotation

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(np.asarray(quaternion, dtype=np.float64)).as_matrix()
    pose[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return pose


def interpolate_poses(pose_a: np.ndarray, pose_b: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate between two rigid poses: lerp position, slerp rotation."""
    from scipy.spatial.transform import Rotation, Slerp

    alpha = float(np.clip(alpha, 0.0, 1.0))
    rotations = Rotation.from_matrix(np.stack([pose_a[:3, :3], pose_b[:3, :3]]))
    slerp = Slerp([0.0, 1.0], rotations)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = slerp([alpha])[0].as_matrix()
    pose[:3, 3] = (1.0 - alpha) * pose_a[:3, 3] + alpha * pose_b[:3, 3]
    return pose


class PoseBuffer:
    """Thread-safe time-indexed buffer of VIO poses.

    A buffer rather than a message-filter synchronizer: VIO runs an order of
    magnitude faster than the localization loop, so by the time a frame is
    processed its stamp is already bracketed by two VIO samples and can be
    interpolated exactly. A synchronizer would instead stall waiting for a pair
    and drop frames whenever one message went missing.
    """

    def __init__(self, max_size: int = 4000) -> None:
        self._lock = threading.Lock()
        self._stamps: List[float] = []
        self._poses: List[np.ndarray] = []
        self.max_size = int(max_size)

    def __len__(self) -> int:
        with self._lock:
            return len(self._stamps)

    def add(self, stamp: float, pose: np.ndarray) -> None:
        stamp = float(stamp)
        pose = np.asarray(pose, dtype=np.float64)
        with self._lock:
            # Usually monotonic, so the common case is an append; a bag replay or
            # a clock jump can go backwards, hence the bisect.
            index = bisect.bisect_right(self._stamps, stamp)
            self._stamps.insert(index, stamp)
            self._poses.insert(index, pose)
            overflow = len(self._stamps) - self.max_size
            if overflow > 0:
                del self._stamps[:overflow]
                del self._poses[:overflow]

    def latest(self) -> Optional[Tuple[float, np.ndarray]]:
        with self._lock:
            if not self._stamps:
                return None
            return self._stamps[-1], self._poses[-1].copy()

    def lookup(self, stamp: float, max_time_diff: float = 0.05) -> Optional[np.ndarray]:
        """The VIO pose at ``stamp``, interpolated between bracketing samples.

        Returns None when the buffer has nothing within ``max_time_diff`` of the
        requested time, which is the honest answer: priming a solve with a pose
        from a different moment is worse than falling back to retrieval.
        """
        stamp = float(stamp)
        with self._lock:
            if not self._stamps:
                return None
            index = bisect.bisect_left(self._stamps, stamp)

            if 0 < index < len(self._stamps):
                before, after = self._stamps[index - 1], self._stamps[index]
                span = after - before
                if span <= 0.0:
                    return self._poses[index].copy()
                if before - max_time_diff <= stamp <= after + max_time_diff:
                    return interpolate_poses(
                        self._poses[index - 1], self._poses[index], (stamp - before) / span
                    )
                return None

            # Outside the buffer: accept the nearest endpoint if it is close
            # enough in time, so a frame stamped just past the newest VIO sample
            # still gets a prior.
            nearest = 0 if index == 0 else len(self._stamps) - 1
            if abs(self._stamps[nearest] - stamp) <= max_time_diff:
                return self._poses[nearest].copy()
            return None

    def clear(self) -> None:
        with self._lock:
            self._stamps.clear()
            self._poses.clear()


class VioAlignment:
    """The map-to-odom transform, re-estimated from each successful fix.

    ``reset_after_failures`` guards against an alignment that has gone bad (VIO
    restarted, tracking lost, a wrong fix): after that many consecutive failed
    VIO-primed attempts the transform is dropped and the next frame goes back
    through retrieval to bootstrap a new one.
    """

    def __init__(self, reset_after_failures: int = 3) -> None:
        self._lock = threading.Lock()
        self._map_from_odom: Optional[np.ndarray] = None
        self.reset_after_failures = max(1, int(reset_after_failures))
        self.consecutive_failures = 0
        self.update_count = 0
        self.last_update_stamp: float = 0.0

    @property
    def is_valid(self) -> bool:
        with self._lock:
            return self._map_from_odom is not None

    @property
    def map_from_odom(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._map_from_odom is None else self._map_from_odom.copy()

    def update(
        self, pose_map_cam: np.ndarray, pose_odom_cam: np.ndarray, stamp: float = 0.0
    ) -> np.ndarray:
        """Re-derive T_map_odom from one localized/VIO pose pair."""
        transform = np.asarray(pose_map_cam, dtype=np.float64) @ invert_pose(pose_odom_cam)
        with self._lock:
            self._map_from_odom = transform
            self.consecutive_failures = 0
            self.update_count += 1
            self.last_update_stamp = float(stamp)
        return transform.copy()

    def predict(self, pose_odom_cam: np.ndarray) -> Optional[np.ndarray]:
        """The camera pose in the map frame implied by a VIO pose."""
        with self._lock:
            if self._map_from_odom is None:
                return None
            return self._map_from_odom @ np.asarray(pose_odom_cam, dtype=np.float64)

    def note_failure(self) -> bool:
        """Count a failed VIO-primed attempt. True if the alignment was dropped."""
        with self._lock:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.reset_after_failures:
                self._map_from_odom = None
                self.consecutive_failures = 0
                return True
            return False

    def invalidate(self) -> None:
        with self._lock:
            self._map_from_odom = None
            self.consecutive_failures = 0

    def status(self) -> dict:
        with self._lock:
            return {
                "aligned": self._map_from_odom is not None,
                "updates": int(self.update_count),
                "consecutive_failures": int(self.consecutive_failures),
                "last_update_stamp": float(self.last_update_stamp),
            }
