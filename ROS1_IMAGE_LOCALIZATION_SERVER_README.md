# ROS1 Image Localization Server

This version keeps MASt3R, the 3DGS model, Gaussian data, pipeline settings, and
reference poses in memory. It replaces folder polling with a ROS subscriber and
publishes the localized camera pose immediately after each successful request.

## Files

- `image_localization_ros1_server.py`: complete ROS1 server.
- `image_localization_server.launch`: launch-file template.

Place the Python script under your catkin package's `scripts/` directory and the
launch file under `launch/`:

```bash
chmod +x scripts/image_localization_ros1_server.py
catkin_make
source devel/setup.bash
```

ROS dependencies normally include:

```bash
sudo apt install ros-noetic-cv-bridge ros-noetic-tf2-ros
```

Your existing Python environment still needs MASt3R, DUSt3R, hloc, PyTorch,
OpenCV, SciPy, and your modified Gaussian Splatting modules.

On this machine that environment is the `gs_loc` conda env (Python 3.10, torch
2.4.1+cu118), which is what the launch files' `python_bin` points at. Note it is
*not* `gaussian_splatting`: that one is Python 3.7 / torch 1.12, the stock 3DGS
training env, and has neither hloc nor the MASt3R dependency stack.

The nodes run under that interpreter through `launch-prefix`, while `PYTHONPATH`
is overridden to ROS' pure-python packages only. Do not launch with the shell's
inherited `PYTHONPATH`: the login profile puts the `hloc2` env's site-packages
first, which shadows `gs_loc`'s numpy and breaks the import.

## Launch

First replace `YOUR_CATKIN_PACKAGE` in the launch file, or pass
`package_name:=...` explicitly.

```bash
roslaunch YOUR_CATKIN_PACKAGE image_localization_server.launch \
  package_name:=YOUR_CATKIN_PACKAGE \
  model_path:=/path/to/3dgs/model \
  database_path:=/media/uwcviss/Expansion/ac1_data/r3live/data_r3live_steeltree_0915 \
  camera_yaml:=/media/uwcviss/Expansion/ac1_data/r3live/data_r3live_steeltree_0915/camera_Raza_staling.yaml \
  image_topic:=/camera/image_raw/compressed \
  compressed:=true
```

Direct `rosrun` is also possible:

```bash
rosrun YOUR_CATKIN_PACKAGE image_localization_ros1_server.py \
  --model_path /path/to/3dgs/model \
  --iteration -1 \
  _database_path:=/path/to/database \
  _camera_yaml:=/path/to/camera.yaml \
  _image_topic:=/camera/image_raw/compressed \
  _compressed:=true
```

## ROS interface

Input:

```text
/camera/image_raw/compressed    sensor_msgs/CompressedImage
```

or, with `compressed:=false`:

```text
/camera/image_raw               sensor_msgs/Image
```

Outputs:

```text
/image_localization/camera_pose geometry_msgs/PoseStamped
/image_localization/status      std_msgs/String (JSON)
/tf                             map -> localized_camera_optical_frame
```

The pose is `T_map_camera`, namely the camera-to-map/world pose. Its position is
the camera center expressed in the map frame.

Inspect results with:

```bash
rostopic echo /image_localization/camera_pose
rostopic echo /image_localization/status
```

A status message contains fields such as:

```json
{
  "success": true,
  "candidate_image": "000123.png",
  "num_matches": 318,
  "num_inliers": 141,
  "inlier_ratio": 0.443,
  "reprojection_rmse_px": 0.84,
  "processing_time_ms": 327.5,
  "queue_delay_ms": 1.3,
  "error": ""
}
```

## Important parameters

- `reference_source=auto`: reference poses come from the COLMAP sparse model the
  map was trained from (`<model_path>/sparse/0`, or `colmap_path`), falling back
  to the model's `cameras.json` when there is none. COLMAP is preferred because
  its world frame is the map frame by construction and it carries the principal
  point and distortion that `cameras.json` drops. `cameras.txt`/`images.txt` are
  read in preference to their `.bin` twins; both work. Force one source with
  `colmap`, `cameras_json` or `loam`.
- `retrieval_source=auto`: a prepared `<database_path>/sfm` descriptor file if
  there is one, else NetVLAD over the map's raw reference images
  (`<model_path>/images` or `reference_images_dir`), else NetVLAD over renders of
  every reference pose. Prefer raw images when you have them: a query is a
  photograph, so photograph-to-photograph retrieval has no render-to-real domain
  gap, and building takes ~40 s per 300 images instead of minutes.
- `camera_yaml` also accepts a COLMAP path, in which case the query intrinsics
  are read from `cameras.txt` instead of a YAML.
- `use_vio=false`: set true when the device also publishes VIO (a drone normally
  does). The first frame still bootstraps through retrieval, because nothing yet
  relates the VIO frame to the map frame; from then on the initial pose is
  predicted from VIO and no retrieval runs. Every new fix re-derives
  `T_map_odom` from the newest pose pair, so VIO drift is corrected rather than
  accumulated, and that transform is published on
  `/image_localization/map_to_odom` so a flight stack can correct the full-rate
  VIO stream between fixes. Set `vio_body_to_camera` if VIO reports a body frame
  rather than the camera optical frame. Leave it false for a device without VIO;
  behaviour is then unchanged. With a VIO prior, drop `optimization_iterations`
  from 3 to 1 — measured on `data/lab`, 310 ms per frame instead of ~1600 ms at
  equal accuracy.
- `reuse_last_pose=false`: every frame uses NetVLAD global retrieval. This is
  safer for independent relocalization requests.
- `reuse_last_pose=true`: first render and match from the previous successful
  pose. NetVLAD retrieval is used only after that attempt fails. This is faster
  for continuous drone video but behaves more like tracking plus relocalization.
- `max_candidates_to_test=1`: preserves the original top-one-candidate logic.
  Increase to 2 or 3 for better robustness at higher latency.
- `optimization_iterations=1`: preserves the original single render/match/PnP
  iteration.
- `runtime_dir=/dev/shm/3dgs_localization`: hloc and MASt3R still require image
  paths in the current codebase, so temporary query/render files are written to
  RAM instead of a normal disk.
- `save_debug=false`: keep this off for normal operation. Enabling it saves the
  query, rendered image, and pose matrix for successful frames.

## Main changes from the benchmark script

1. The 3DGS scene and MASt3R model are loaded once during startup.
2. Folder scanning, `sleep()`, and text-file pose output are removed.
3. The ROS callback never runs GPU localization; it only stores the latest
   message.
4. Pending images use a one-element buffer, preventing old frames from queuing.
5. PnP receives the correct world-to-camera initial guess.
6. Only matched depth pixels are backprojected, rather than allocating 3D points
   for the entire rendered image.
7. Incoming image resolution can differ from the camera YAML; intrinsics are
   scaled automatically.
8. The server survives individual localization exceptions and publishes the
   failure reason on the status topic.

## Validation before flight

Verify these conventions with a known reference image before connecting the
pose to control:

- the published transform is camera-to-map, not map-to-camera;
- the camera optical frame follows the expected ROS axis convention;
- the YAML intrinsics correspond to the actual ROS image after any crop,
  resize, or undistortion;
- the 3DGS map frame is the same frame expected by the planner/controller;
- if the controller needs `base_link`, apply the calibrated static camera-to-body
  transform using TF2 rather than treating the camera pose as the drone pose.
