#!/usr/bin/env python3
"""Publishes offline images to the localization server and reports the results.

Reads query images from a folder, publishes them on the server's image topic at a
fixed rate, and subscribes to the pose and status topics so each frame's outcome
is printed as it comes back.

The camera intrinsics come from the same YAML the server loads (``~camera_yaml``,
default ``config/test_camera.yaml``), so there is one place to enter a
calibration. They are also published as ``sensor_msgs/CameraInfo`` for anything
else listening.

    rosrun YOUR_PACKAGE test_image_publisher.py \
        _image_dir:=data/colmap_E2/test_image \
        _camera_yaml:=config/test_camera.yaml

``~image_names`` publishes a named subset of ``~image_dir`` instead of all of
it; ``~stride`` and ``~max_images`` thin a large folder. Replaying reference
images only measures something if the server was told to keep those same frames
out of its retrieval database (``~retrieval_exclude``), otherwise retrieval hands
back the query itself and the solve starts from the answer.

With ``~ground_truth`` pointing at a COLMAP sparse model (or a dataset root
holding one, or a trained model's ``cameras.json``), any query whose filename
matches a reference image is scored against that pose, so the test reports
actual localization error rather than just success.

VIO simulation
--------------
``~publish_vio:=true`` also publishes a *simulated* VIO stream derived from those
same ground-truth poses: an arbitrary odom origin plus drift that accumulates per
frame (``~vio_drift_m_per_frame``, ``~vio_drift_deg_per_frame``). It exists to
exercise the server's ``~use_vio`` path on a dataset, with no drone in the room.
It is not a VIO implementation and has no use outside testing.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Dict, List, Optional

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_config import load_camera_config  # noqa: E402

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def split_list(value) -> List[str]:
    """Parse a comma- or whitespace-separated ROS param into a list of strings."""
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).replace(",", " ").split() if part.strip()]


def list_images(
    folder: str, names: Optional[List[str]] = None, stride: int = 1, limit: int = 0
) -> List[str]:
    """Query image paths, optionally an explicit subset of the folder.

    ``names`` selects specific files, which is how a test replays a handful of
    frames the server was told to keep out of its retrieval database. Otherwise
    the whole folder is taken, thinned by ``stride`` and capped at ``limit``.
    """
    if not os.path.isdir(folder):
        raise RuntimeError("Image folder does not exist: {}".format(folder))

    if names:
        paths = []
        for name in names:
            path = name if os.path.isabs(name) else os.path.join(folder, name)
            if not os.path.isfile(path):
                raise RuntimeError("Requested test image does not exist: {}".format(path))
            paths.append(path)
        return paths

    found = sorted(
        name for name in os.listdir(folder) if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not found:
        raise RuntimeError(
            "No images in {} (looked for {})".format(folder, ", ".join(IMAGE_EXTENSIONS))
        )
    found = found[:: max(1, int(stride))]
    if limit and limit > 0:
        found = found[: int(limit)]
    return [os.path.join(folder, name) for name in found]


def load_ground_truth(path: str) -> Dict[str, np.ndarray]:
    """Load c2w poses keyed by image name.

    ``path`` is either a COLMAP sparse model (or a root containing one) or a 3DGS
    ``cameras.json``. Keys are reduced to the filename stem so a query saved as
    ``.jpg`` still scores against a reference listed as ``.png``.
    """
    if not path:
        return {}

    if os.path.isdir(path) or not path.lower().endswith(".json"):
        from colmap_model import find_model_dir, read_model

        found = find_model_dir(path)
        if found is None:
            rospy.logwarn("No COLMAP model under %s; ground-truth scoring is off", path)
            return {}
        poses = read_model(found).poses_c2w()
        return {os.path.splitext(os.path.basename(k))[0]: v for k, v in poses.items()}

    if not os.path.isfile(path):
        return {}
    with open(path, "r") as handle:
        entries = json.load(handle)
    poses = {}
    for entry in entries:
        pose = np.eye(4)
        pose[:3, :3] = np.asarray(entry["rotation"], dtype=np.float64)
        pose[:3, 3] = np.asarray(entry["position"], dtype=np.float64)
        poses[os.path.splitext(str(entry["img_name"]))[0]] = pose
    return poses


class SimulatedVio:
    """A fake VIO stream built from ground-truth poses, for testing only.

    Two properties of real VIO are reproduced, because they are the two the
    server has to cope with:

    * an arbitrary origin — the odom frame is unrelated to the map frame, so the
      server has to estimate the map-to-odom transform rather than assume it;
    * drift — a slowly growing error, so a transform estimated at frame k is
      already slightly wrong at frame k+1 and has to be re-estimated.

    Poses are published as ``nav_msgs/Odometry`` in a burst around each image
    stamp, at ``rate_hz``, so the server's buffer has samples bracketing the
    image and genuinely interpolates rather than picking the nearest one.
    """

    def __init__(
        self,
        origin_xyz_rpy=(11.0, -4.0, 2.5, -12.0, 5.0, 37.0),
        drift_m_per_frame: float = 0.02,
        drift_deg_per_frame: float = 0.3,
        rate_hz: float = 20.0,
        frame_id: str = "odom",
        child_frame_id: str = "vio_body",
    ) -> None:
        from scipy.spatial.transform import Rotation

        self.odom_from_map = np.eye(4)
        self.odom_from_map[:3, :3] = Rotation.from_euler(
            "xyz", origin_xyz_rpy[3:6], degrees=True
        ).as_matrix()
        self.odom_from_map[:3, 3] = origin_xyz_rpy[0:3]
        self.drift_m_per_frame = float(drift_m_per_frame)
        self.drift_deg_per_frame = float(drift_deg_per_frame)
        self.rate_hz = max(1.0, float(rate_hz))
        self.frame_id = frame_id
        self.child_frame_id = child_frame_id
        self.frame_index = 0

    def _drift(self, elapsed_frames: float) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        drift = np.eye(4)
        drift[:3, :3] = Rotation.from_euler(
            "z", self.drift_deg_per_frame * elapsed_frames, degrees=True
        ).as_matrix()
        drift[:3, 3] = [self.drift_m_per_frame * elapsed_frames, 0.0, 0.0]
        return drift

    def pose_in_odom(self, pose_map_cam: np.ndarray, elapsed_frames: float) -> np.ndarray:
        return self._drift(elapsed_frames) @ self.odom_from_map @ np.asarray(pose_map_cam)

    def burst(
        self,
        pose_previous: np.ndarray,
        pose_current: np.ndarray,
        stamp,
        span: float = 0.2,
    ) -> List[Odometry]:
        """VIO samples running from the previous frame's pose up to this one.

        The samples end just past the image stamp so the server's buffer has
        entries on both sides of it and interpolates, which is what a real VIO
        stream running faster than the camera looks like.
        """
        from scipy.spatial.transform import Rotation

        from vio_prior import interpolate_poses

        count = max(2, int(round(span * self.rate_hz)))
        messages = []
        for step in range(count + 2):  # one extra, just past the image stamp
            alpha = min(1.0, step / float(count))
            offset = -span + span * step / float(count)
            pose_map_cam = interpolate_poses(pose_previous, pose_current, alpha)
            pose = self.pose_in_odom(pose_map_cam, self.frame_index - 1.0 + alpha)
            quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
            message = Odometry()
            message.header.stamp = stamp + rospy.Duration(offset)
            message.header.frame_id = self.frame_id
            message.child_frame_id = self.child_frame_id
            message.pose.pose.position.x = float(pose[0, 3])
            message.pose.pose.position.y = float(pose[1, 3])
            message.pose.pose.position.z = float(pose[2, 3])
            message.pose.pose.orientation.x = float(quaternion[0])
            message.pose.pose.orientation.y = float(quaternion[1])
            message.pose.pose.orientation.z = float(quaternion[2])
            message.pose.pose.orientation.w = float(quaternion[3])
            messages.append(message)
        self.frame_index += 1
        return messages


def pose_message_to_matrix(message: PoseStamped) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    pose = np.eye(4)
    orientation = message.pose.orientation
    pose[:3, :3] = Rotation.from_quat(
        [orientation.x, orientation.y, orientation.z, orientation.w]
    ).as_matrix()
    pose[:3, 3] = [message.pose.position.x, message.pose.position.y, message.pose.position.z]
    return pose


class TestImagePublisher:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = rospy.get_param(
            "~image_dir", os.path.join(here, "data", "colmap_E2", "test_image")
        )
        camera_yaml = rospy.get_param("~camera_yaml", os.path.join(here, "config", "test_camera.yaml"))
        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw/compressed")
        self.use_compressed = bool(rospy.get_param("~compressed", True))
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/camera_info")
        self.frame_id = rospy.get_param("~frame_id", "camera_optical_frame")
        self.rate_hz = float(rospy.get_param("~rate", 0.2))
        self.repeat = int(rospy.get_param("~repeat", 1))
        self.startup_delay = float(rospy.get_param("~startup_delay", 5.0))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 95))
        self.wait_for_result = bool(rospy.get_param("~wait_for_result", True))
        self.result_timeout = float(rospy.get_param("~result_timeout", 120.0))

        self.camera_config = load_camera_config(camera_yaml)
        rospy.loginfo("Test publisher camera: %s", self.camera_config.describe())

        # ~ground_truth_json is the older name and still works.
        self.ground_truth = load_ground_truth(
            rospy.get_param("~ground_truth", "") or rospy.get_param("~ground_truth_json", "")
        )

        self.publish_vio = bool(rospy.get_param("~publish_vio", False))
        self.vio_simulator: Optional[SimulatedVio] = None
        self.vio_publisher = None
        self._previous_truth: Optional[np.ndarray] = None
        if self.publish_vio:
            if not self.ground_truth:
                raise RuntimeError(
                    "~publish_vio needs ~ground_truth: the simulated VIO stream is "
                    "derived from the reference poses"
                )
            self.vio_simulator = SimulatedVio(
                drift_m_per_frame=float(rospy.get_param("~vio_drift_m_per_frame", 0.02)),
                drift_deg_per_frame=float(rospy.get_param("~vio_drift_deg_per_frame", 0.3)),
                rate_hz=float(rospy.get_param("~vio_rate", 20.0)),
                frame_id=rospy.get_param("~vio_frame", "odom"),
            )
            self.vio_publisher = rospy.Publisher(
                rospy.get_param("~vio_topic", "/vio/odometry"), Odometry, queue_size=200
            )
            rospy.logwarn(
                "Publishing a SIMULATED VIO stream on %s (drift %.3f m + %.2f deg per "
                "frame). For testing the server's ~use_vio path only.",
                rospy.get_param("~vio_topic", "/vio/odometry"),
                self.vio_simulator.drift_m_per_frame,
                self.vio_simulator.drift_deg_per_frame,
            )
        if self.ground_truth:
            rospy.loginfo("Loaded %d ground-truth poses", len(self.ground_truth))

        self.image_paths = list_images(
            self.image_dir,
            names=split_list(rospy.get_param("~image_names", "")),
            stride=int(rospy.get_param("~stride", 1)),
            limit=int(rospy.get_param("~max_images", 0)),
        )
        rospy.loginfo("Publishing %d test images from %s", len(self.image_paths), self.image_dir)

        message_type = CompressedImage if self.use_compressed else Image
        self.image_publisher = rospy.Publisher(self.image_topic, message_type, queue_size=1)
        self.camera_info_publisher = rospy.Publisher(
            self.camera_info_topic, CameraInfo, queue_size=1, latch=True
        )
        rospy.Subscriber(
            rospy.get_param("~pose_topic", "/image_localization/camera_pose"),
            PoseStamped,
            self._pose_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            rospy.get_param("~status_topic", "/image_localization/status"),
            String,
            self._status_callback,
            queue_size=10,
        )

        self._result_event = threading.Event()
        self._latest_pose: Optional[np.ndarray] = None
        self._current_name = ""
        self._succeeded = 0
        self._attempted = 0
        self._errors: List[float] = []
        self._rotation_errors: List[float] = []
        self._init_sources: Dict[str, int] = {}

    # -- callbacks -------------------------------------------------------

    def _pose_callback(self, message: PoseStamped) -> None:
        self._latest_pose = pose_message_to_matrix(message)

    def _status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except ValueError:
            rospy.logwarn("Could not parse status message")
            return

        if status.get("success"):
            self._succeeded += 1
            timings = status.get("timings_ms", {})
            source = status.get("init_source", "")
            self._init_sources[source] = self._init_sources.get(source, 0) + 1
            rospy.loginfo(
                "  -> OK   %s | init=%s candidate=%s inliers=%d/%d rmse=%.2fpx %.0fms %s",
                self._current_name,
                source or "?",
                status.get("candidate_image", "?"),
                status.get("num_inliers", 0),
                status.get("num_matches", 0),
                status.get("reprojection_rmse_px") or float("nan"),
                status.get("processing_time_ms", 0.0),
                " ".join("{}={}".format(k.replace("_ms", ""), v) for k, v in sorted(timings.items())),
            )
            vio = status.get("vio")
            if vio:
                rospy.loginfo(
                    "         vio: aligned=%s updates=%d prediction_error=%s",
                    vio.get("aligned"),
                    vio.get("updates", 0),
                    "{:.4f} m".format(vio["prediction_error_m"])
                    if "prediction_error_m" in vio
                    else "-",
                )
            pose = status.get("pose_c2w")
            if pose is not None:
                position = np.asarray(pose)[:3, 3]
                rospy.loginfo(
                    "         position [%.3f %.3f %.3f]", position[0], position[1], position[2]
                )
                self._score_against_ground_truth(np.asarray(pose))
        else:
            rospy.logwarn("  -> FAIL %s | %s", self._current_name, status.get("error", ""))
        self._result_event.set()

    def _score_against_ground_truth(self, estimated: np.ndarray) -> None:
        reference = self.ground_truth.get(os.path.splitext(self._current_name)[0])
        if reference is None:
            return
        translation = float(np.linalg.norm(estimated[:3, 3] - reference[:3, 3]))
        relative = estimated[:3, :3].T @ reference[:3, :3]
        cosine = (np.trace(relative) - 1.0) / 2.0
        rotation = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        self._errors.append(translation)
        self._rotation_errors.append(rotation)
        rospy.loginfo("         error vs ground truth: %.4f m, %.4f deg", translation, rotation)

    # -- publishing ------------------------------------------------------

    def _camera_info(self, width: int, height: int, stamp) -> CameraInfo:
        config = self.camera_config.scaled_to(width, height)
        camera = config.camera
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.frame_id
        info.width = width
        info.height = height
        info.distortion_model = (
            "plumb_bob" if config.distortion_model == "none" else config.distortion_model
        )
        info.D = [float(value) for value in config.distortion_coeffs] or [0.0] * 5
        info.K = [camera.fx, 0.0, camera.cx, 0.0, camera.fy, camera.cy, 0.0, 0.0, 1.0]
        info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.P = [
            camera.fx, 0.0, camera.cx, 0.0,
            0.0, camera.fy, camera.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return info

    def _publish_simulated_vio(self, stamp) -> None:
        """Emit the VIO burst for the frame about to be published.

        Sent *before* the image so the server's buffer already brackets the
        image stamp when the worker looks it up.
        """
        if self.vio_simulator is None or self.vio_publisher is None:
            return
        truth = self.ground_truth.get(os.path.splitext(self._current_name)[0])
        if truth is None:
            rospy.logwarn_throttle(
                5.0,
                "No ground-truth pose for %s, so no simulated VIO for this frame",
                self._current_name,
            )
            return
        previous = self._previous_truth if self._previous_truth is not None else truth
        for message in self.vio_simulator.burst(previous, truth, stamp):
            self.vio_publisher.publish(message)
        self._previous_truth = truth
        # Give the server's callback a moment to drain the burst before the
        # image arrives; otherwise the first lookup can race the buffer.
        rospy.sleep(0.05)

    def _publish_one(self, path: str) -> bool:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            rospy.logwarn("Could not read %s, skipping", path)
            return False

        height, width = image.shape[:2]
        stamp = rospy.Time.now()
        self._current_name = os.path.basename(path)
        self._result_event.clear()
        self._attempted += 1
        rospy.loginfo("[%d] publishing %s (%dx%d)", self._attempted, self._current_name, width, height)

        self.camera_info_publisher.publish(self._camera_info(width, height, stamp))
        self._publish_simulated_vio(stamp)

        if self.use_compressed:
            ok, encoded = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if not ok:
                rospy.logwarn("Could not JPEG-encode %s", path)
                return False
            message = CompressedImage()
            message.header.stamp = stamp
            message.header.frame_id = self.frame_id
            message.format = "jpeg"
            message.data = encoded.tobytes()
        else:
            message = Image()
            message.header.stamp = stamp
            message.header.frame_id = self.frame_id
            message.height = height
            message.width = width
            message.encoding = "bgr8"
            message.is_bigendian = 0
            message.step = width * 3
            message.data = image.tobytes()

        self.image_publisher.publish(message)
        return True

    def run(self) -> None:
        # The server spends a while loading the map, MASt3R and the retrieval
        # database. Publishing into a subscriber-less topic would just drop
        # frames, so wait for it to connect.
        deadline = rospy.Time.now() + rospy.Duration(self.startup_delay)
        while (
            not rospy.is_shutdown()
            and self.image_publisher.get_num_connections() == 0
            and rospy.Time.now() < deadline
        ):
            rospy.sleep(0.2)
        if self.image_publisher.get_num_connections() == 0:
            rospy.logwarn(
                "No subscriber on %s after %.1fs; publishing anyway",
                self.image_topic,
                self.startup_delay,
            )

        period = 1.0 / self.rate_hz if self.rate_hz > 0 else 0.0
        for _ in range(max(1, self.repeat)):
            for path in self.image_paths:
                if rospy.is_shutdown():
                    break
                if not self._publish_one(path):
                    continue
                if self.wait_for_result:
                    if not self._result_event.wait(timeout=self.result_timeout):
                        rospy.logwarn(
                            "  -> no result within %.0fs for %s",
                            self.result_timeout,
                            self._current_name,
                        )
                elif period > 0:
                    rospy.sleep(period)

        self._summary()

    def _summary(self) -> None:
        rospy.loginfo("=" * 64)
        rospy.loginfo(
            "localized %d/%d frames", self._succeeded, self._attempted
        )
        if self._errors:
            rospy.loginfo(
                "pose error vs ground truth: median %.4f m / %.4f deg over %d frames",
                float(np.median(self._errors)),
                float(np.median(self._rotation_errors)),
                len(self._errors),
            )
        if self._init_sources:
            rospy.loginfo(
                "initial pose came from: %s",
                ", ".join(
                    "{} x{}".format(source or "?", count)
                    for source, count in sorted(self._init_sources.items())
                ),
            )
        rospy.loginfo("=" * 64)


def main() -> None:
    sys.argv = rospy.myargv(argv=sys.argv)
    rospy.init_node("test_image_publisher", anonymous=True)
    try:
        TestImagePublisher().run()
    except Exception as exc:
        rospy.logfatal("test_image_publisher failed: %s", exc)
        raise
    if bool(rospy.get_param("~shutdown_when_done", True)):
        rospy.signal_shutdown("test finished")
    else:
        rospy.spin()


if __name__ == "__main__":
    main()
