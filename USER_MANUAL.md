# GS-CPE Image Relocalization Server — User Manual

A ROS1 node that takes a single camera image and returns the camera's 6-DoF pose
in the frame of a pre-built 3D Gaussian Splatting map. No prior pose, no odometry
and no marker is needed: every frame is relocalized independently.

```
 sensor_msgs/Image or CompressedImage
              |
              v
   NetVLAD retrieval  ->  nearest reference view  ->  its pose as a starting guess
              |
              v
   render RGB + depth from the 3DGS map at that pose
              |
              v
   MASt3R dense 2D-2D matching  (render <-> query)
              |
              v
   backproject matched render pixels through the rendered depth  ->  3D points
              |
              v
   cv2.solvePnPRansac (2D-3D)  ->  refined pose      [optionally iterated]
              |
              v
   geometry_msgs/PoseStamped  +  JSON status  +  /tf
```

Measured on an RTX 3070 Ti Laptop with the bundled `data/colmap_E2` map at
1224x1024: **~2.0 s per frame** (match 1.28 s, PnP 0.44 s, retrieval 0.20 s,
render 0.10 s), 683/3225 inliers, 1.82 px reprojection RMSE.

---

## 1. Requirements

| | |
| --- | --- |
| OS | Ubuntu 20.04 |
| ROS | Noetic |
| GPU | NVIDIA, CUDA 11.8 capable, **>= 6 GB VRAM** (colmap_E2 uses ~3.9 GB) |
| CUDA toolkit | 11.8, with `nvcc` on `PATH` — needed to *compile* the 3DGS rasterizer |
| Compiler | gcc/g++ 9 (Ubuntu 20.04 default) |
| Disk | ~15 GB for the env, plus the map |

Verified on driver 535.183.01, gcc 9.4.0, CUDA 11.8.89.

---

## 2. Building the `gs_loc` conda env from scratch

> The env is called `gs_loc`. It is **not** the same as a stock `gaussian_splatting`
> training env — that one is Python 3.7 / torch 1.12 and cannot run this pipeline
> (MASt3R needs torch 2.x, and hloc needs Python >= 3.8).

### 2.1 Create the env

```bash
conda create -n gs_loc python=3.10 -y
conda activate gs_loc
```

### 2.2 PyTorch (must match your CUDA toolkit)

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.4.1+cu118 True
```

### 2.3 Python dependencies

```bash
pip install \
  numpy==1.26.4 \
  opencv-python==4.10.0.84 \
  scipy pyyaml h5py matplotlib tqdm pillow plyfile \
  einops roma safetensors huggingface_hub gdown
```

`numpy` must stay on the 1.x line — several of the pinned wheels below are built
against the NumPy 1 ABI and will fail at import under NumPy 2.

### 2.4 The 3DGS rasterizer

Build it from the `gaussian-splatting/` checkout that ships inside this
repository. Make sure `nvcc --version` reports 11.8 first.

```bash
cd /path/to/3DGS_Loc_server/gaussian-splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
# submodules/fused-ssim is only needed for *training* a map; skip it.
```

Check:

```bash
python -c "import diff_gaussian_rasterization, simple_knn._C; print('rasterizer OK')"
```

If the build fails with a gcc/nvcc version complaint, export
`export TORCH_CUDA_ARCH_LIST="8.6"` (use your GPU's compute capability) and retry.

### 2.5 hloc (for NetVLAD retrieval)

Stock upstream hloc is sufficient — the localization server does **not** need the
private fork. (Only the older `gs_cpe_mast3r_server_Raza*.py` benchmark scripts
call the fork's custom `save_global_candidates_for_query`.)

```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization.git
cd Hierarchical-Localization
pip install -e .
pip install pycolmap==0.6.1     # hloc 1.5 does not work with newer pycolmap
python -c "from hloc import extract_features; print('hloc OK')"
```

### 2.6 MASt3R / DUSt3R

Both are **already vendored** in this repository (`mast3r/`, `dust3r/`) and
resolve locally — nothing to install. `mast3r/utils/path_to_dust3r.py` puts
`dust3r` on `sys.path`, so keep the two directories as siblings.

Model weights are pulled from HuggingFace on first startup and cached in
`~/.cache/huggingface`:
`naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` (~2 GB, needs internet
once).

*Optional speedup:* without a compiled RoPE2D kernel you will see
`Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch
version instead`. It is harmless. To build it:

```bash
cd /path/to/3DGS_Loc_server/dust3r/croco/models/curope
python setup.py build_ext --inplace
```

### 2.7 ROS bindings

`rospy` is pure Python, so the conda interpreter can use ROS' own packages
directly — no ROS rebuild is required. Nothing to install here; the launch files
put `/opt/ros/noetic/lib/python3/dist-packages` on `PYTHONPATH` for you.

`cv_bridge` and `tf2_ros` are **compiled** against the system Python 3.8 and will
not import under this env. That is expected and handled: the node falls back to
its own image decoder and publishes `/tf` as a raw `TFMessage`.

### 2.8 Verify the whole stack

```bash
conda activate gs_loc
cd /path/to/3DGS_Loc_server
env -u PYTHONPATH python selftest_localization.py --model_path data/colmap_E2
```

This runs the geometry chain offline with no ROS. Expect all pairs to solve with
a median error of a few centimetres. If this passes, the pipeline is sound and
anything that fails later is ROS plumbing or calibration.

### 2.9 The one environment gotcha

Many setups export a `PYTHONPATH` in `~/.bashrc` (e.g. pointing at another conda
env's `site-packages`). That shadows `gs_loc`'s own packages and produces:

```
ImportError: Error importing numpy: ... No module named 'numpy.core._multiarray_umath'
```

Run scripts with `env -u PYTHONPATH python ...`, or `unset PYTHONPATH` first. The
launch files already override `PYTHONPATH` per node, so `roslaunch` is unaffected.

---

## 3. Data you need

### 3.1 The map (`model_path`)

A trained 3DGS model directory. The complete layout, as in `data/lab/`:

```
<model_path>/
  point_cloud/iteration_30000/point_cloud.ply   <- the map itself
  sparse/0/cameras.txt                          <- reference intrinsics
  sparse/0/images.txt                           <- reference poses
  images/                                       <- raw reference images
```

`sparse/0` is the COLMAP reconstruction the map was trained from, so its world
frame *is* the map frame. That is the preferred source of reference poses: it is
written by a tool outside this repository, and unlike `cameras.json` it carries
the principal point and the distortion model. `.txt` and `.bin` are both read,
text preferred when both exist. To produce the text form from a binary model:

```bash
colmap model_converter --input_path sparse/0 --output_path sparse/0 --output_type TXT
```

Inspect what the server will see with:

```bash
env -u PYTHONPATH python colmap_model.py data/lab
```

`reference_source` selects where poses come from:

| value | meaning |
| --- | --- |
| `auto` (default) | COLMAP model if the map has one, else `cameras.json` |
| `colmap` | require a COLMAP model; fail if there is none |
| `cameras_json` | the trained model's own `cameras.json` |
| `loam` | `<database_path>/loam/0/poses.csv` via the Gaussian_splatting fork |

Point `colmap_path:=` at the sparse model directory when it does not sit under
`model_path`. Searched automatically: `.`, `sparse/0`, `sparse`,
`colmap/sparse/0`, `colmap/sparse`, `sfm/sparse/0`, `sparse/txt`.

`data/colmap_E2/` is the older example: a trained model with a `cameras.json` and
no reconstruction or source images on this machine, so `auto` falls back to
`cameras.json` there. It still works; it just cannot use the two COLMAP features
above.

### 3.2 The retrieval database

`retrieval_source` selects how the NetVLAD database is built:

| value | meaning | cost |
| --- | --- | --- |
| `auto` (default) | prepared h5, else raw images, else renders | — |
| `images` | NetVLAD over the map's raw reference images | ~40 s / 300 images |
| `render` | NetVLAD over a render of every reference pose | minutes |
| `folder` | only a prepared `<database_path>/sfm/global-feats-netvlad.h5` | none |

**Prefer `images` when you have the source photographs.** A query frame is a
photograph, so comparing it against photographs rather than against renders
removes a domain gap that costs retrieval accuracy, and it skips the rendering
pass. `render` exists for maps whose source images are not on the machine, which
is the case for `data/colmap_E2`.

The images are found at `<model_path>/images` or next to the COLMAP model;
override with `reference_images_dir:=`. Only images that have a reference pose
enter the database — a retrieval hit is useless without a pose to start from.

Either build is cached in `<model_path>/netvlad_cache/` (override with
`retrieval_cache_dir:=`). The cache manifest records the source and a fingerprint
of the file set, so switching `retrieval_source` or changing the images rebuilds
rather than silently reusing stale descriptors. Force a rebuild with
`rebuild_retrieval_db:=true` (needed if you retrain the map).

`retrieval_exclude:="a.png,b.png"` keeps named images out of the database. This
is for offline testing: replaying a reference image as a query measures nothing
if that same image is in the database, because retrieval returns it and the solve
starts from the answer.

### 3.3 Camera intrinsics (`camera_yaml`)

Your measured calibration for the query camera. Copy `config/test_camera.yaml`
and edit:

```yaml
cam_width:  1224
cam_height: 1024
cam_fx: 912.194
cam_fy: 911.791
cam_cx: 634.692
cam_cy: 513.263

# Distortion of the image you actually publish. Use "none" if you publish
# already-rectified images (e.g. from image_proc), otherwise it is corrected twice.
#   none | plumb_bob (k1,k2,p1,p2,k3) | equidistant (fisheye, exactly k1..k4)
distortion_model: none
distortion_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]
```

Plain names (`fx`, `width`, `image_width`, ...) are accepted as well as the
`cam_*` LOAM/r3live flavour, and the distortion model accepts aliases
(`radtan`/`radial_tangential` for `plumb_bob`, `fisheye`/`kannala_brandt` for
`equidistant`, `pinhole` for `none`), so most existing calibration files work
unedited.

If your calibration comes from COLMAP, do not retype it. `camera_yaml` accepts a
COLMAP path directly, so this works with no YAML at all:

```bash
roslaunch image_localization_3dgs image_localization_server.launch \
  model_path:=/path/to/data/lab camera_yaml:=/path/to/data/lab/sparse/0
```

To capture it as a YAML you can then edit (e.g. to declare distortion COLMAP
modelled as pinhole):

```python
from camera_config import load_colmap_camera, save_camera_config
save_camera_config(load_colmap_camera("data/lab/sparse/0"), "config/my_camera.yaml")
```

`config/lab_camera.yaml` was made this way and is the query camera for the
`data/lab` map.

Enter the principal point as measured — do not round `cam_cx`/`cam_cy` to the
image centre. If incoming images arrive at a different resolution than
`cam_width` x `cam_height`, the server rescales `fx, fy, cx, cy` automatically and
logs a warning; it cannot detect a *crop*, so prefer calibrating at the
resolution you publish.

---

## 4. Running the server

Put this directory on `ROS_PACKAGE_PATH`, or symlink it into `~/catkin_ws/src`
and `catkin_make`. Without a catkin build:

```bash
export ROS_PACKAGE_PATH=/path/to/3DGS_Loc_server:/opt/ros/noetic/share
```

Then:

```bash
roslaunch image_localization_3dgs image_localization_server.launch \
  model_path:=/path/to/3dgs/model \
  camera_yaml:=/path/to/your_camera.yaml \
  image_topic:=/camera/image_raw/compressed \
  compressed:=true
```

Edit `python_bin` in the launch file if your conda env is not at
`$HOME/miniconda3/envs/gs_loc`.

The server is ready when it logs:

```
Localization engine ready
Listening on /camera/image_raw/compressed (CompressedImage); publishing poses on /image_localization/camera_pose
```

---

## 5. ROS interface

### Subscribed

| Topic | Type | Notes |
| --- | --- | --- |
| `~image_topic` (default `/camera/image_raw/compressed`) | `sensor_msgs/CompressedImage` if `~compressed:=true`, else `sensor_msgs/Image` | queue size 1 |
| `~vio_topic` (default `/vio/odometry`) | `nav_msgs/Odometry` by default; see `~vio_msg_type` | only if `~use_vio:=true` |

Raw `Image` encodings supported: `mono8`, `8UC1`, `bgr8`, `rgb8`, `8UC3`,
`bgra8`, `rgba8`, `8UC4`.

The subscriber callback does **no** GPU work — it only writes the newest message
into a one-element slot. A single worker thread does all localization, so if
frames arrive faster than ~0.5 Hz the stale ones are dropped rather than queued.
The server always works on the freshest available frame.

### Published

| Topic | Type | Notes |
| --- | --- | --- |
| `~pose_topic` (default `/image_localization/camera_pose`) | `geometry_msgs/PoseStamped` | published only on success |
| `~status_topic` (default `/image_localization/status`) | `std_msgs/String` | JSON, published on **every** frame incl. failures |
| `/image_localization/map_to_odom` | `geometry_msgs/PoseStamped` | the map-to-odom transform, latched; only if `~use_vio` |
| `/tf` | `tf2_msgs/TFMessage` | `map_frame` -> `camera_frame`, if `~publish_tf`; plus `map_frame` -> `~vio_frame` if `~publish_map_to_odom_tf` |

The pose header carries the **incoming image's** stamp (or `now()` if the
incoming stamp is zero), so results stay aligned with the source frames.

### Pose convention

The published pose is **T_map_camera (camera-to-world)**:

- `position` is the camera centre in the map frame.
- `orientation` rotates the camera optical frame (x right, y down, z forward)
  into the map frame.

This is what you want for RViz and for a `map -> camera` TF. If you need the
`solvePnP`-style world-to-camera extrinsic, invert it.

### Status JSON

```json
{
  "success": true,
  "stamp": 1785953590.89,
  "candidate_image": "1590192710.099499.png",
  "num_matches": 3225,
  "num_inliers": 683,
  "inlier_ratio": 0.212,
  "reprojection_rmse_px": 1.82,
  "processing_time_ms": 2018.0,
  "queue_delay_ms": 0.4,
  "timings_ms": {"render_ms": 101.2, "match_ms": 1276.5, "pnp_ms": 443.8, "retrieval_ms": 195.9},
  "received_count": 1, "processed_count": 1, "dropped_count": 0,
  "error": "",
  "pose_c2w": [[...], [...], [...], [...]]
}
```

`init_source` says where the initial pose came from: `retrieval`, `vio` or
`previous_pose`. With `~use_vio` there is also a `vio` block:

```json
"vio": {"aligned": true, "updates": 15, "consecutive_failures": 0,
        "last_update_stamp": 1786454292.7, "buffered_poses": 340,
        "frames_without_pose": 0, "prediction_error_m": 0.0732}
```

Watch this topic first when debugging — exceptions inside the pipeline are caught
and reported here rather than killing the node:

```bash
rostopic echo /image_localization/status
```

---

## 6. Using VIO instead of retrieval

If the device already runs VIO — a drone almost always does — its odometry can
supply the initial pose, and NetVLAD retrieval then runs only on the first frame
and whenever the alignment is lost.

```bash
roslaunch image_localization_3dgs image_localization_server.launch \
  model_path:=/path/to/data/lab camera_yaml:=config/lab_camera.yaml \
  use_vio:=true vio_topic:=/vins_estimator/odometry
```

`~use_vio:=false` (the default) leaves the server exactly as it was, for a device
with no VIO.

### How it works

VIO is locally accurate but drifts and starts at an arbitrary origin, so its
poses live in their own frame (*odom*), related to the map by one rigid
transform:

```
T_map_camera(t) = T_map_odom @ T_odom_camera(t)
```

`T_map_odom` is unknown until the first successful fix, which does use retrieval.
After that one pose pair it is known, and every later frame is primed by
predicting `T_map_odom @ T_odom_camera(t)` — no retrieval. **Each new fix
re-derives `T_map_odom` from the newest pose pair**, never integrating, so VIO
drift is corrected rather than accumulated.

The order of attempts per frame is: VIO prior → previous pose (if
`~reuse_last_pose`) → retrieval. If a VIO-primed solve fails, the frame falls
back to retrieval (`~vio_fallback_to_retrieval`), and after
`~vio_reset_after_failures` consecutive failures the alignment is dropped so the
next frame re-bootstraps.

### Synchronization

The two streams do **not** have to be synchronized message-for-message. VIO poses
are buffered and interpolated (slerp + lerp) to each image stamp, which is what a
VIO running faster than the camera looks like. They do have to share a clock: a
frame with no VIO pose within `~vio_max_time_diff` (default 50 ms) falls back to
retrieval and logs a throttled warning. Check `frames_without_pose` in the status
JSON if you suspect a clock offset.

### The body-to-camera extrinsic

VIO normally reports a body/IMU frame, not the camera optical frame. Set
`~vio_body_to_camera` to `T_body_camera` as `x y z qx qy qz qw` (or 16 row-major
matrix values). A constant error here is absorbed into `T_map_odom` and does
**not** bias the result globally, but it does corrupt the *relative* prediction
in proportion to how much the body rotates between fixes — so on a drone that
yaws, enter it.

### Feeding the correction back

`/image_localization/map_to_odom` carries the current `T_map_odom` (latched).
That is the useful output for a flight stack: compose it with the full-rate VIO
pose to get a drift-corrected map-frame pose at VIO rate, between the ~1-3 Hz
fixes. `~publish_map_to_odom_tf:=true` publishes the same thing as a
`map -> ~vio_frame` TF; it is off by default because it claims a parent frame
that an existing VIO tf tree may already own.

### What it actually saves

Retrieval is only ~60 ms of a ~1500 ms frame, so skipping it is not the win. The
win is that a VIO prior lands within a few centimetres, and from there **one**
render/match/PnP pass is as accurate as three. Measured on `data/lab`, 16
consecutive frames:

| | init | `optimization_iterations` | median error | per frame |
| --- | --- | --- | --- | --- |
| retrieval | every frame | 3 | 3.2 mm / 0.20° | ~1500 ms |
| VIO | 1st frame only | 3 | 4.5 mm / 0.18° | ~1600 ms |
| VIO | 1st frame only | 1 | **4.2 mm / 0.14°** | **~310 ms** |

So set `optimization_iterations:=1` when using VIO. The accuracy comes from the
prior, not from re-iterating.

### Testing it without a drone

```bash
roslaunch image_localization_3dgs test_localization_vio.launch
```

The test publisher emits a *simulated* VIO stream derived from the ground-truth
poses (arbitrary origin, 2 cm + 0.3°/frame drift, 20 Hz against 2 Hz images).
Expect `init=retrieval` on frame 1 and `init=vio` on the rest, and a
`prediction_error` that settles around one frame's worth of drift instead of
growing:

```
localized 16/16 frames
pose error vs ground truth: median 0.0042 m / 0.1375 deg over 16 frames
initial pose came from: retrieval x1, vio x15
```

`~publish_vio` is a test fixture only — it is not a VIO implementation and has no
use outside replaying a dataset.

---

## 7. Testing with offline images

`test_image_publisher.py` reads images from a folder, publishes them on the
server's image topic, and prints each result as it comes back. Both nodes at
once:

```bash
roslaunch image_localization_3dgs test_localization.launch
```

Defaults to the bundled `data/colmap_E2` map and `data/colmap_E2/test_image/`.
Expected output:

```
[1] publishing 1590192597.699922.png (1224x1024)
  -> OK   1590192597.699922.png | candidate=1590192710.099499.png inliers=683/3225 rmse=1.82px 2018ms
         position [1.557 -0.172 2.982]
localized 1/1 frames
```

For the `data/lab` map — COLMAP reference poses and a retrieval database built
from the raw images:

```bash
roslaunch image_localization_3dgs test_localization_lab.launch
```

It replays six frames spread along the trajectory and holds those same frames
out of the retrieval database (`holdout_frames`), so every solve has to start
from a different reference view. Measured on this machine:

```
localized 6/6 frames
pose error vs ground truth: median 0.0019 m / 0.1794 deg over 6 frames
```

Startup is ~12 s with a warm cache (~50 s the first time, while NetVLAD runs
over the 302 database images), and ~1.5 s per frame at 1882x1058.

Against an already-running server:

```bash
env -u PYTHONPATH python -u test_image_publisher.py \
  _image_dir:=/path/to/images \
  _camera_yaml:=config/test_camera.yaml
```

Useful publisher params: `~rate`, `~repeat`, `~wait_for_result` (wait for each
result before sending the next — keeps the log readable), `~result_timeout`,
`~startup_delay`, `~compressed`. To publish part of a large folder: `~image_names`
(an explicit comma-separated subset), `~stride`, `~max_images`.

**Scoring against ground truth.** Set `~ground_truth` to a COLMAP sparse model
(or a root holding one) or to a `cameras.json`; any query whose *filename stem*
matches a reference image is scored and the median translation/rotation error is
printed in the summary. A genuinely held-out test image will not match, and no
error line is printed — that is correct, not a failure. (`data/colmap_E2`'s test
frame is one: it was an eval frame, so it is not in `cameras.json`.)

**Replaying reference images.** If the query is itself a reference image, tell
the server to keep it out of the retrieval database with `~retrieval_exclude`, as
`test_localization_lab.launch` does. Otherwise retrieval returns the query
itself, the solve starts from the true pose, and the reported error means
nothing.

---

## 8. Tuning

Everything below is a ROS private param; see
`launch/image_localization_server.launch` for the full annotated list.

| Param | Default | Meaning |
| --- | --- | --- |
| `reference_source` | `auto` | Where reference poses come from: `auto`, `colmap`, `cameras_json`, `loam`. See §3.1. |
| `colmap_path` | (empty) | Explicit COLMAP sparse model directory, when it is not under `model_path`. |
| `retrieval_source` | `auto` | How the NetVLAD database is built: `auto`, `images`, `render`, `folder`. See §3.2. |
| `reference_images_dir` | (empty) | Raw reference images, when they are not at `<model_path>/images`. |
| `retrieval_exclude` | (empty) | Comma-separated images to keep out of the database. Offline testing only. |
| `optimization_iterations` | 3 | Re-render at the current estimate and re-solve. Biggest accuracy lever; cost scales linearly. |
| `num_retrieval` | 3 | NetVLAD candidates fetched. |
| `max_candidates_to_test` | 1 | How many are actually tried. Raise to 2-3 for robustness in repetitive scenes, at proportional cost. |
| `reuse_last_pose` | false | `true` = try the previous pose before retrieval. Faster, but becomes tracking-with-relocalization-fallback rather than independent relocalization. |
| `match_subsample` | 8 | MASt3R match stride. Lower = more correspondences, slower. |
| `pnp_reprojection_error_px` | 3.0 | RANSAC inlier threshold. |
| `min_inliers` | 12 | Below this a frame is reported as failed. |
| `depth_mode` | `inverse` | The rasterizer returns expected *inverse* depth. Only change if you swap in a rasterizer returning metric z. |
| `render_use_principal_point` | false | Stock 3DGS builds cameras from focal lengths alone, so maps trained that way must be rendered with a centred principal point. Set true only for a fork that genuinely models cx/cy. |
| `save_debug` / `debug_dir` | false | Dump query, render, depth and pose per frame. |
| `runtime_dir` | `/dev/shm/3dgs_localization` | Scratch files for hloc/MASt3R. Keep on tmpfs. |

---

## 9. Troubleshooting

**`ImportError: numpy.core._multiarray_umath`** — a stale `PYTHONPATH` from your
shell profile. See §2.9.

**`RLException: Invalid roslaunch XML syntax`** — you have a `--` inside an XML
comment in a launch file. It is illegal in XML; reword it.

**The node prints nothing for minutes after "Reusing cached NetVLAD database"** —
if you removed the `-u` from `launch-prefix`, Python buffers stdout when
roslaunch captures it through a pipe and an already-running node looks hung.
Confirm with `rostopic info <image_topic>` — if the server is listed as a
subscriber, it is up. Keep the `-u`.

**First startup takes several minutes** — it is building the NetVLAD database
(one time) and downloading MASt3R weights (one time). Watch the log for
`rendered N/260`.

**Every frame fails with "Too few 2D-3D correspondences"** — usually the query
camera is looking at something the map does not cover, or the intrinsics are
wrong. Check `candidate_image` in the status topic and compare that reference
view against your query by eye. Run `selftest_localization.py` to confirm the
map/geometry side is healthy independent of your camera.

**Poses are consistently offset by a fixed amount** — suspect the principal
point. See `render_use_principal_point` in §8.

**CUDA out of memory** — lower the published image resolution, or reduce
`db_render_scale` (database build only).

---

## 10. File map

| File | Role |
| --- | --- |
| `image_localization_ros1_server.py` | The ROS1 node — glue only |
| `localization_pipeline.py` | The pipeline (retrieval -> render -> match -> PnP), ROS-free |
| `gs_render_backend.py` | 3DGS map loading and RGB+depth rendering |
| `mast3r_matching.py` | MASt3R dense matching and pixel back-mapping |
| `netvlad_retrieval.py` | NetVLAD extractor + database build/cache |
| `colmap_model.py` | COLMAP sparse-model reader (text and binary); also a CLI to inspect one |
| `camera_config.py` | Camera YAML/COLMAP loading, rescaling, undistortion |
| `selftest_localization.py` | Offline geometry self-test, no ROS |
| `test_image_publisher.py` | Offline image publisher + result scorer |
| `launch/image_localization_server.launch` | Server only |
| `launch/test_localization.launch` | Server + test publisher (colmap_E2 defaults) |
| `launch/test_localization_lab.launch` | The same, with data/lab arguments |
| `config/test_camera.yaml` | Query camera intrinsics, colmap_E2 |
| `config/lab_camera.yaml` | Query camera intrinsics, data/lab |
