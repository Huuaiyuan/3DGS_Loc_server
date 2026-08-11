#!/usr/bin/env python3
"""ROS1 online image-localization server.

    sensor_msgs/Image or sensor_msgs/CompressedImage
                         |
                         v
           NetVLAD retrieval -> nearest reference pose
                         |
                         v
           3DGS RGB + depth rendering (persistent model)
                         |
                         v
              MASt3R dense 2D-2D correspondences
                         |
                         v
                    PnP-RANSAC
                         |
                         v
              geometry_msgs/PoseStamped

This module is only the ROS glue; the pipeline itself lives in
``localization_pipeline.py`` so it can also be run offline (see
``selftest_localization.py``).

The subscriber callback never localizes, it only stores the newest message. One
worker thread processes the newest available frame, so images do not queue up
behind a running solve.

Pose convention
---------------
The published pose is T_map_camera (camera-to-world): its translation is the
camera centre in the map frame, and its rotation maps the camera optical frame
into the map frame.

Reference poses
---------------
``~reference_source=auto`` (default) prefers the COLMAP sparse model the map was
trained from — ``<model_path>/sparse/0`` or ``~colmap_path``, text encoding
preferred over binary — and falls back to the model's ``cameras.json`` when there
is none. Force one with ``colmap`` or ``cameras_json``; ``loam`` keeps the
original behaviour and reads ``<database_path>/loam/0/poses.csv`` through the
private Gaussian_splatting fork.

Retrieval database
------------------
``~retrieval_source=auto`` (default) takes a prepared
``<database_path>/sfm/global-feats-netvlad.h5`` if there is one, otherwise builds
a database from the map's raw reference images (``<model_path>/images`` or
``~reference_images_dir``), otherwise renders every reference pose. Either build
is cached under ``~retrieval_cache_dir``.

VIO priming
-----------
``~use_vio=true`` subscribes to the device's VIO stream (``~vio_topic``) and uses
it, rather than retrieval, to produce the initial pose. Retrieval then runs only
on the first frame and whenever the alignment is lost. The VIO poses are buffered
and interpolated to each image stamp, so the two streams do not have to be
synchronized message-for-message; they only have to share a clock.

The estimated map-to-odom transform is published on
``/image_localization/map_to_odom``, which is what lets a consumer correct the
full-rate VIO stream between fixes. Default off, so a device without VIO behaves
exactly as before.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from argparse import ArgumentParser
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# cv_bridge is a compiled extension built against the system python3.8. Running
# this node from a conda env with a different interpreter is the normal case
# here, so image decoding falls back to a small pure-numpy path.
try:
    from cv_bridge import CvBridge, CvBridgeError

    _CV_BRIDGE_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the interpreter in use
    CvBridge = None
    CvBridgeError = Exception
    _CV_BRIDGE_AVAILABLE = False

from camera_config import CameraConfig, load_camera_config, load_colmap_camera  # noqa: E402
from gs_render_backend import DEFAULT_GS_REPO  # noqa: E402
from localization_pipeline import (  # noqa: E402
    MAST3R_DEFAULT,
    LocalizationEngine,
    LocalizationResult,
)
from vio_prior import PoseBuffer, pose_from_position_quaternion  # noqa: E402

# What ~vio_msg_type may be set to, and the message each one names. Odometry is
# the default because that is what VINS-Fusion, OpenVINS and the ORB-SLAM3 ROS
# wrappers publish.
VIO_MESSAGE_TYPES = ("odometry", "pose", "pose_cov", "transform")

_ENCODING_CHANNELS = {
    "mono8": 1,
    "8UC1": 1,
    "bgr8": 3,
    "rgb8": 3,
    "8UC3": 3,
    "bgra8": 4,
    "rgba8": 4,
    "8UC4": 4,
}


def imgmsg_to_bgr(message: Image) -> np.ndarray:
    """Decode sensor_msgs/Image to BGR without cv_bridge."""
    channels = _ENCODING_CHANNELS.get(message.encoding)
    if channels is None:
        raise ValueError(
            "Unsupported image encoding '{}'. Supported: {}".format(
                message.encoding, ", ".join(sorted(_ENCODING_CHANNELS))
            )
        )
    buffer = np.frombuffer(message.data, dtype=np.uint8)
    expected = message.height * message.step
    if buffer.size < expected:
        raise ValueError(
            "Truncated image: {} bytes, expected {}".format(buffer.size, expected)
        )
    array = buffer[:expected].reshape(message.height, message.step)
    array = array[:, : message.width * channels].reshape(
        message.height, message.width, channels
    )
    if message.encoding == "rgb8":
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    if message.encoding == "rgba8":
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    if message.encoding == "bgra8":
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    if channels == 1:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return array.copy()


def _split_list(value: Any) -> list:
    """Parse a comma- or whitespace-separated ROS param into a list of strings."""
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).replace(",", " ").split() if part.strip()]


def parse_extrinsic(value: Any) -> np.ndarray:
    """Parse ``~vio_body_to_camera`` into a 4x4 transform.

    Accepts ``[x, y, z, qx, qy, qz, qw]`` or a flat 16-element row-major matrix.
    """
    values = _split_list(value)
    if not values:
        return np.eye(4, dtype=np.float64)
    numbers = [float(item) for item in values]
    if len(numbers) == 7:
        return pose_from_position_quaternion(numbers[:3], numbers[3:])
    if len(numbers) == 16:
        return np.asarray(numbers, dtype=np.float64).reshape(4, 4)
    raise ValueError(
        "vio_body_to_camera needs 7 values [x y z qx qy qz qw] or 16 matrix "
        "values, got {}".format(len(numbers))
    )


def load_query_camera(path: str) -> CameraConfig:
    """Load the query intrinsics from a calibration YAML or a COLMAP model.

    A YAML is the normal case (it is the one place a measured calibration is
    entered). Accepting a COLMAP path as well means a camera that was itself
    calibrated by COLMAP can be pointed at directly, with no transcription step.
    """
    if str(path).lower().endswith((".yaml", ".yml")):
        return load_camera_config(path)
    return load_colmap_camera(path)


def compressed_to_bgr(message: CompressedImage) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cv2.imdecode returned None for the compressed image")
    return image


class LatestImageSlot:
    """Thread-safe one-element buffer that always keeps the newest frame."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: Optional[Tuple[Any, float]] = None
        self.received_count = 0
        self.processed_count = 0
        self.dropped_count = 0

    def put(self, message: Any) -> None:
        with self._condition:
            if self._latest is not None:
                self.dropped_count += 1
            self._latest = (message, time.perf_counter())
            self.received_count += 1
            self._condition.notify()

    def get(self, timeout: float = 0.5) -> Optional[Tuple[Any, float]]:
        with self._condition:
            if self._latest is None:
                self._condition.wait(timeout=timeout)
            if self._latest is None:
                return None
            item = self._latest
            self._latest = None
            self.processed_count += 1
            return item


class ImageLocalizationRosNode:
    def __init__(self) -> None:
        self.slot = LatestImageSlot()
        self.shutdown_event = threading.Event()
        self.bridge = CvBridge() if _CV_BRIDGE_AVAILABLE else None
        if not _CV_BRIDGE_AVAILABLE:
            rospy.loginfo("cv_bridge unavailable; using the built-in image decoder")

        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw/compressed")
        self.use_compressed = bool(rospy.get_param("~compressed", True))
        self.pose_topic = rospy.get_param("~pose_topic", "/image_localization/camera_pose")
        self.status_topic = rospy.get_param("~status_topic", "/image_localization/status")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.camera_frame = rospy.get_param("~camera_frame", "localized_camera_optical_frame")
        self.publish_tf = bool(rospy.get_param("~publish_tf", True))
        self.publish_map_to_odom_tf = bool(rospy.get_param("~publish_map_to_odom_tf", False))

        model_path = self._require_param("~model_path")
        camera_config = load_query_camera(self._require_param("~camera_yaml"))
        rospy.loginfo("Query camera: %s", camera_config.describe())

        self.use_vio = bool(rospy.get_param("~use_vio", False))
        self.vio_max_time_diff = float(rospy.get_param("~vio_max_time_diff", 0.05))
        self.vio_frame = rospy.get_param("~vio_frame", "odom")
        self.body_to_camera = parse_extrinsic(rospy.get_param("~vio_body_to_camera", ""))
        self.vio_buffer = PoseBuffer(int(rospy.get_param("~vio_buffer_size", 4000)))
        self.vio_missing_count = 0

        default_runtime = (
            "/dev/shm/3dgs_localization"
            if os.path.isdir("/dev/shm")
            else "/tmp/3dgs_localization"
        )

        self.engine = LocalizationEngine(
            model_path=model_path,
            camera_config=camera_config,
            runtime_dir=rospy.get_param("~runtime_dir", default_runtime),
            iteration=int(rospy.get_param("~iteration", -1)),
            database_path=rospy.get_param("~database_path", ""),
            reference_source=rospy.get_param("~reference_source", "auto"),
            colmap_path=rospy.get_param("~colmap_path", ""),
            gs_repo_path=rospy.get_param("~gs_repo_path", DEFAULT_GS_REPO),
            mast3r_model=rospy.get_param("~mast3r_model", MAST3R_DEFAULT),
            device=rospy.get_param("~device", "cuda"),
            sh_degree=int(rospy.get_param("~sh_degree", 3)),
            white_background=bool(rospy.get_param("~white_background", False)),
            depth_mode=rospy.get_param("~depth_mode", "inverse"),
            render_use_principal_point=bool(
                rospy.get_param("~render_use_principal_point", False)
            ),
            retrieval_source=rospy.get_param("~retrieval_source", "auto"),
            reference_images_dir=rospy.get_param("~reference_images_dir", ""),
            retrieval_exclude=_split_list(rospy.get_param("~retrieval_exclude", "")),
            retrieval_cache_dir=rospy.get_param("~retrieval_cache_dir", ""),
            rebuild_retrieval_db=bool(rospy.get_param("~rebuild_retrieval_db", False)),
            db_render_scale=float(rospy.get_param("~db_render_scale", 0.5)),
            num_retrieval=int(rospy.get_param("~num_retrieval", 3)),
            max_candidates_to_test=int(rospy.get_param("~max_candidates_to_test", 1)),
            optimization_iterations=int(rospy.get_param("~optimization_iterations", 3)),
            min_correspondences=int(rospy.get_param("~min_correspondences", 20)),
            min_inliers=int(rospy.get_param("~min_inliers", 12)),
            pnp_iterations=int(rospy.get_param("~pnp_iterations", 2000)),
            pnp_reprojection_error_px=float(rospy.get_param("~pnp_reprojection_error_px", 3.0)),
            pnp_confidence=float(rospy.get_param("~pnp_confidence", 0.9999)),
            refine_pnp_lm=bool(rospy.get_param("~refine_pnp_lm", True)),
            min_depth=float(rospy.get_param("~min_depth", 1e-3)),
            max_depth=float(rospy.get_param("~max_depth", 1e4)),
            match_subsample=int(rospy.get_param("~match_subsample", 8)),
            reuse_last_pose=bool(rospy.get_param("~reuse_last_pose", False)),
            use_vio=self.use_vio,
            vio_reset_after_failures=int(rospy.get_param("~vio_reset_after_failures", 3)),
            vio_fallback_to_retrieval=bool(
                rospy.get_param("~vio_fallback_to_retrieval", True)
            ),
            jpeg_quality=int(rospy.get_param("~jpeg_quality", 95)),
            save_debug=bool(rospy.get_param("~save_debug", False)),
            debug_dir=rospy.get_param("~debug_dir", "/tmp/3dgs_localization_debug"),
            logger=rospy.loginfo,
            warner=rospy.logwarn,
        )

        self.pose_publisher = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.status_publisher = rospy.Publisher(self.status_topic, String, queue_size=10)
        # Publishing TFMessage directly avoids tf2_ros, whose compiled tf2_py
        # extension is built for the system python3.8 only.
        self.tf_publisher = (
            rospy.Publisher("/tf", TFMessage, queue_size=10) if self.publish_tf else None
        )

        self.map_to_odom_publisher = rospy.Publisher(
            rospy.get_param("~map_to_odom_topic", "/image_localization/map_to_odom"),
            PoseStamped,
            queue_size=1,
            latch=True,
        )

        message_type = CompressedImage if self.use_compressed else Image
        self.subscriber = rospy.Subscriber(
            self.image_topic,
            message_type,
            self._image_callback,
            queue_size=1,
            buff_size=int(rospy.get_param("~subscriber_buffer_bytes", 2 ** 26)),
            tcp_nodelay=True,
        )

        self.vio_subscriber = self._subscribe_vio() if self.use_vio else None

        self.worker = threading.Thread(
            target=self._worker_loop, name="image_localization_worker", daemon=True
        )
        self.worker.start()
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo(
            "Listening on %s (%s); publishing poses on %s",
            self.image_topic,
            "CompressedImage" if self.use_compressed else "Image",
            self.pose_topic,
        )

    @staticmethod
    def _require_param(name: str) -> str:
        value = rospy.get_param(name, "")
        if not value:
            raise rospy.ROSInitException("Required parameter {} is not set".format(name))
        return value

    # -- VIO -------------------------------------------------------------

    def _subscribe_vio(self):
        """Subscribe to the VIO stream under whichever message type it uses."""
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from nav_msgs.msg import Odometry

        topic = rospy.get_param("~vio_topic", "/vio/odometry")
        kind = str(rospy.get_param("~vio_msg_type", "odometry")).strip().lower()
        types = {
            "odometry": Odometry,
            "pose": PoseStamped,
            "pose_cov": PoseWithCovarianceStamped,
            "transform": TransformStamped,
        }
        if kind not in types:
            raise rospy.ROSInitException(
                "~vio_msg_type must be one of {}, got '{}'".format(
                    ", ".join(VIO_MESSAGE_TYPES), kind
                )
            )
        rospy.loginfo(
            "VIO priming on: %s (%s), body->camera extrinsic %s",
            topic,
            kind,
            "identity" if np.allclose(self.body_to_camera, np.eye(4)) else "set",
        )
        return rospy.Subscriber(
            topic, types[kind], self._vio_callback, queue_size=200, tcp_nodelay=True
        )

    def _vio_callback(self, message: Any) -> None:
        """Buffer one VIO sample. No GPU work, no blocking."""
        try:
            if hasattr(message, "transform"):
                translation = message.transform.translation
                orientation = message.transform.rotation
            else:
                pose = message.pose
                pose = pose.pose if hasattr(pose, "pose") else pose  # PoseWithCovariance
                translation = pose.position
                orientation = pose.orientation
            stamp = message.header.stamp
            self.vio_buffer.add(
                (stamp if stamp != rospy.Time() else rospy.Time.now()).to_sec(),
                pose_from_position_quaternion(
                    [translation.x, translation.y, translation.z],
                    [orientation.x, orientation.y, orientation.z, orientation.w],
                ),
            )
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Could not read a VIO message: %s", exc)

    def _vio_pose_for(self, stamp) -> Optional[np.ndarray]:
        """The camera pose in the VIO frame at an image stamp, or None."""
        if not self.use_vio:
            return None
        pose_odom_body = self.vio_buffer.lookup(stamp.to_sec(), self.vio_max_time_diff)
        if pose_odom_body is None:
            self.vio_missing_count += 1
            rospy.logwarn_throttle(
                5.0,
                "No VIO pose within %.3f s of the image stamp (%d frames so far); "
                "this frame falls back to retrieval. Check that both streams use "
                "the same clock.",
                self.vio_max_time_diff,
                self.vio_missing_count,
            )
            return None
        # VIO reports a body frame; the pipeline localizes the camera.
        return pose_odom_body @ self.body_to_camera

    def _publish_map_to_odom(self, stamp) -> None:
        transform = self.engine.vio_alignment.map_from_odom
        if transform is None:
            return
        quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.map_frame
        message.pose.position.x = float(transform[0, 3])
        message.pose.position.y = float(transform[1, 3])
        message.pose.position.z = float(transform[2, 3])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        self.map_to_odom_publisher.publish(message)

        # Off by default: a map -> odom TF is the standard way to express this,
        # but it claims a parent that an existing VIO tree may already own.
        if self.tf_publisher is not None and self.publish_map_to_odom_tf:
            transform_message = TransformStamped()
            transform_message.header = message.header
            transform_message.child_frame_id = self.vio_frame
            transform_message.transform.translation.x = message.pose.position.x
            transform_message.transform.translation.y = message.pose.position.y
            transform_message.transform.translation.z = message.pose.position.z
            transform_message.transform.rotation = message.pose.orientation
            self.tf_publisher.publish(TFMessage(transforms=[transform_message]))

    def _image_callback(self, message: Any) -> None:
        # No GPU work here: a new message simply replaces the pending one.
        self.slot.put(message)

    def _decode_image(self, message: Any) -> np.ndarray:
        if self.use_compressed:
            return compressed_to_bgr(message)
        if self.bridge is not None:
            try:
                return self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            except CvBridgeError as exc:
                raise ValueError("cv_bridge conversion failed: {}".format(exc))
        return imgmsg_to_bgr(message)

    def _worker_loop(self) -> None:
        while not rospy.is_shutdown() and not self.shutdown_event.is_set():
            item = self.slot.get(timeout=0.5)
            if item is None:
                continue
            message, arrival_time = item
            queue_delay_ms = (time.perf_counter() - arrival_time) * 1000.0
            stamp = message.header.stamp
            if stamp == rospy.Time():
                stamp = rospy.Time.now()

            try:
                vio_pose = self._vio_pose_for(stamp)
                result = self.engine.localize(
                    self._decode_image(message), vio_pose, stamp.to_sec()
                )
            except Exception as exc:
                result = LocalizationResult(
                    success=False, error="{}: {}".format(type(exc).__name__, exc)
                )
                rospy.logerr(traceback.format_exc())

            if result.success and result.pose_c2w is not None:
                self._publish_pose(result.pose_c2w, stamp)
                if self.use_vio:
                    self._publish_map_to_odom(stamp)
                position = result.pose_c2w[:3, 3]
                rospy.loginfo(
                    "Frame %d localized at [%.3f %.3f %.3f] via %s: "
                    "%d/%d inliers, RMSE %.2f px, %.0f ms",
                    self.slot.processed_count,
                    position[0],
                    position[1],
                    position[2],
                    result.candidate_image,
                    result.num_inliers,
                    result.num_matches,
                    result.reprojection_rmse_px,
                    result.processing_time_ms,
                )
            else:
                rospy.logwarn("Frame %d failed: %s", self.slot.processed_count, result.error)
            self._publish_status(result, stamp, queue_delay_ms)

    def _publish_pose(self, pose_c2w: np.ndarray, stamp) -> None:
        quaternion = Rotation.from_matrix(pose_c2w[:3, :3]).as_quat()
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.map_frame
        message.pose.position.x = float(pose_c2w[0, 3])
        message.pose.position.y = float(pose_c2w[1, 3])
        message.pose.position.z = float(pose_c2w[2, 3])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        self.pose_publisher.publish(message)

        if self.tf_publisher is not None:
            transform = TransformStamped()
            transform.header = message.header
            transform.child_frame_id = self.camera_frame
            transform.transform.translation.x = message.pose.position.x
            transform.transform.translation.y = message.pose.position.y
            transform.transform.translation.z = message.pose.position.z
            transform.transform.rotation = message.pose.orientation
            self.tf_publisher.publish(TFMessage(transforms=[transform]))

    def _publish_status(self, result: LocalizationResult, stamp, queue_delay_ms: float) -> None:
        payload = {
            "success": bool(result.success),
            "stamp": stamp.to_sec(),
            "candidate_image": result.candidate_image,
            "init_source": result.init_source,
            "num_matches": int(result.num_matches),
            "num_inliers": int(result.num_inliers),
            "inlier_ratio": float(result.inlier_ratio),
            "reprojection_rmse_px": (
                float(result.reprojection_rmse_px)
                if np.isfinite(result.reprojection_rmse_px)
                else None
            ),
            "processing_time_ms": float(result.processing_time_ms),
            "queue_delay_ms": float(queue_delay_ms),
            "timings_ms": {k: round(float(v), 2) for k, v in result.timings_ms.items()},
            "received_count": int(self.slot.received_count),
            "processed_count": int(self.slot.processed_count),
            "dropped_count": int(self.slot.dropped_count),
            "error": result.error,
        }
        if result.success and result.pose_c2w is not None:
            payload["pose_c2w"] = [[float(v) for v in row] for row in np.asarray(result.pose_c2w)]
        if self.use_vio:
            vio = self.engine.vio_alignment.status()
            vio["buffered_poses"] = len(self.vio_buffer)
            vio["frames_without_pose"] = int(self.vio_missing_count)
            if np.isfinite(result.vio_prediction_error_m):
                vio["prediction_error_m"] = float(result.vio_prediction_error_m)
            payload["vio"] = vio
        self.status_publisher.publish(String(data=json.dumps(payload)))

    def shutdown(self) -> None:
        self.shutdown_event.set()


def main() -> None:
    # roslaunch appends remapping arguments such as __name:=... that argparse
    # cannot parse; rospy.myargv strips them.
    sys.argv = rospy.myargv(argv=sys.argv)
    parser = ArgumentParser(description="ROS1 3DGS image localization server")
    parser.add_argument("--model_path", "-m", default=None)
    parser.add_argument("--iteration", default=None, type=int)
    args, _ = parser.parse_known_args()

    rospy.init_node("image_localization_server", anonymous=False)
    # Command-line values win over private params, so both invocation styles work.
    if args.model_path:
        rospy.set_param("~model_path", args.model_path)
    if args.iteration is not None:
        rospy.set_param("~iteration", args.iteration)

    try:
        ImageLocalizationRosNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logfatal("Failed to start the localization server: %s", exc)
        rospy.logfatal(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
