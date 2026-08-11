#!/usr/bin/env python3
"""NetVLAD global retrieval for the localization server.

Three ways to get a retrieval database:

``folder``
    The original setup: a prepared database directory holding ``images/`` and
    precomputed descriptors at ``sfm/global-feats-netvlad.h5``.

``images``
    Extract descriptors from the map's own *raw* reference images, the ones
    COLMAP was run on (``data/lab/images``). Preferred whenever they are
    available: a query frame is a real photograph, so comparing it against real
    photographs rather than against renders removes the domain gap that costs
    retrieval accuracy, and it skips the rendering pass entirely.

``render``
    Build one from the 3DGS map itself, by rendering every reference pose and
    extracting NetVLAD descriptors from those renders. Needs nothing beyond the
    trained model, which is what makes it work with ``data/colmap_E2`` — a
    trained model whose source images are not on this machine.

Either built database is cached, so the cost is paid once per map.

Queries go through a persistent extractor. ``hloc.extract_features.main()`` is
built for batch jobs: it constructs the NetVLAD model, walks a directory and
writes an HDF5 file on every call, which measured ~4.5 s per frame here. Holding
the model and the database descriptors in memory instead brings that to the
forward pass alone.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

CACHE_MANIFEST = "manifest.json"
DESCRIPTOR_NAME = "global-feats-netvlad.h5"


def import_hloc():
    """Import hloc under whichever name is installed.

    The benchmark scripts import ``Hierarchical_Localization.hloc``; a normal
    editable install of the same repository exposes it as ``hloc``.
    """
    try:
        from hloc import extract_features, pairs_from_retrieval

        return extract_features, pairs_from_retrieval
    except ImportError as exc:
        # Only fall through when hloc itself is absent. A failure *inside* hloc
        # (a missing dependency of its own, such as pycolmap) must not be
        # reported as "Hierarchical_Localization not found".
        if getattr(exc, "name", "") not in ("hloc", None):
            raise

    from Hierarchical_Localization.hloc import extract_features, pairs_from_retrieval

    return extract_features, pairs_from_retrieval


def _hloc_submodule(extract_features_module, name: str):
    """Import a sibling module of the resolved hloc package."""
    package = extract_features_module.__name__.rsplit(".", 1)[0]
    return importlib.import_module("{}.{}".format(package, name))


def normalize_candidates(raw: Any, query_name: str) -> List[str]:
    """Normalize the several shapes hloc helpers return."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        values = raw.get(query_name, raw.get(os.path.basename(query_name), []))
        return [values] if isinstance(values, str) else [str(v) for v in values]
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, np.ndarray):
        return [str(item) for item in raw.tolist()]
    if isinstance(raw, Sequence):
        return [str(item) for item in raw]
    return [str(raw)]


def preprocess_for_netvlad(image_bgr: np.ndarray, resize_max: int = 1024) -> torch.Tensor:
    """Reproduce hloc's ImageDataset preprocessing for a single in-memory image.

    It has to match exactly, or query descriptors would not be comparable with
    the database ones: RGB, float32, long side capped at ``resize_max`` with area
    interpolation, CHW, scaled to [0, 1].
    """
    image = image_bgr[:, :, ::-1].astype(np.float32)  # BGR -> RGB
    height, width = image.shape[:2]
    if resize_max and max(width, height) > resize_max:
        scale = resize_max / float(max(width, height))
        new_size = (int(round(width * scale)), int(round(height * scale)))
        interpolation = cv2.INTER_AREA
        if new_size[0] > width or new_size[1] > height:
            interpolation = cv2.INTER_LINEAR
        image = cv2.resize(image, new_size, interpolation=interpolation)
    image = image.transpose((2, 0, 1)) / 255.0
    return torch.from_numpy(np.ascontiguousarray(image)).float()[None]


class NetVLADRetrieval:
    """Persistent NetVLAD extractor plus an in-memory descriptor database."""

    def __init__(
        self,
        db_descriptor_path: str,
        work_dir: str,
        device: str = "cuda",
        logger=None,
    ) -> None:
        self.db_descriptor_path = os.path.abspath(db_descriptor_path)
        if not os.path.isfile(self.db_descriptor_path):
            raise FileNotFoundError(
                "NetVLAD database descriptors not found: {}".format(self.db_descriptor_path)
            )
        self.work_dir = os.path.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self._log = logger or (lambda *a: None)

        self._extract_features, self._pairs_from_retrieval = import_hloc()
        self.conf = self._extract_features.confs["netvlad"]
        self.resize_max = int(self.conf["preprocessing"].get("resize_max") or 0)

        self._model = self._load_model()
        self._db_names, self._db_descriptors = self._load_database()
        self._log(
            "Retrieval ready: %d database descriptors of dimension %d",
            len(self._db_names),
            self._db_descriptors.shape[1],
        )

    def _load_model(self):
        extractors = _hloc_submodule(self._extract_features, "extractors")
        base_model = _hloc_submodule(self._extract_features, "utils.base_model")
        model_class = base_model.dynamic_load(extractors, self.conf["model"]["name"])
        return model_class(self.conf["model"]).eval().to(self.device)

    def _load_database(self) -> Tuple[List[str], torch.Tensor]:
        import h5py

        io_module = _hloc_submodule(self._extract_features, "utils.io")
        names = list(io_module.list_h5_names(self.db_descriptor_path))
        if not names:
            raise RuntimeError(
                "No descriptors in {}".format(self.db_descriptor_path)
            )
        with h5py.File(self.db_descriptor_path, "r", libver="latest") as handle:
            descriptors = [
                np.asarray(handle[name]["global_descriptor"]).reshape(-1) for name in names
            ]
        stacked = torch.from_numpy(np.stack(descriptors)).float().to(self.device)
        return names, stacked

    @torch.no_grad()
    def retrieve(self, image_bgr: np.ndarray, num_matched: int = 3) -> List[str]:
        """Return the closest database image names for a BGR query image."""
        tensor = preprocess_for_netvlad(image_bgr, self.resize_max).to(self.device)
        prediction = self._model({"image": tensor})
        descriptor = prediction["global_descriptor"].reshape(1, -1).float()

        similarity = descriptor @ self._db_descriptors.t()
        count = int(min(max(1, num_matched), len(self._db_names)))
        indices = torch.topk(similarity.reshape(-1), count).indices.tolist()
        return [self._db_names[index] for index in indices]


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _manifest_for(
    model_path: str, iteration: int, width: int, height: int, num_poses: int
) -> Dict[str, Any]:
    return {
        "source": "render",
        "model_path": os.path.abspath(model_path),
        "iteration": int(iteration),
        "width": int(width),
        "height": int(height),
        "num_poses": int(num_poses),
        "descriptor": DESCRIPTOR_NAME,
    }


def _cache_is_current(
    manifest: Dict[str, Any], descriptor_path: str, manifest_path: str, log
) -> bool:
    if not os.path.isfile(descriptor_path) or not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path, "r") as handle:
            cached = json.load(handle)
    except (ValueError, IOError):
        return False
    if isinstance(cached, dict) and "source" not in cached:
        # Manifests written before retrieval had more than one source; they only
        # ever described renders. Filling it in avoids re-rendering a whole map
        # to reproduce descriptors that are already correct.
        cached["source"] = "render"
    if cached == manifest:
        log("Reusing cached NetVLAD database: %s", descriptor_path)
        return True
    log("Cached NetVLAD database does not match this map, rebuilding")
    return False


def _extract_netvlad(image_dir: str, names: Sequence[str], cache_dir: str, log) -> str:
    """Run hloc's NetVLAD extractor over ``names`` and return the h5 path."""
    extract_features, _ = import_hloc()
    log("Extracting NetVLAD descriptors for %d database images", len(names))
    produced = str(
        extract_features.main(
            conf=extract_features.confs["netvlad"],
            image_dir=Path(image_dir),
            export_dir=cache_dir,
            image_list=list(names),
            overwrite=True,
        )
    )
    descriptor_path = os.path.join(cache_dir, DESCRIPTOR_NAME)
    if os.path.abspath(produced) != os.path.abspath(descriptor_path):
        os.replace(produced, descriptor_path)
    return descriptor_path


def _stems(names) -> set:
    return {os.path.splitext(os.path.basename(str(name)))[0] for name in names}


def list_reference_images(
    image_dir: str,
    reference_poses: Optional[Dict[str, np.ndarray]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """Image files in ``image_dir`` that have a reference pose.

    Retrieval is only useful when the returned name can be turned into an
    initial pose, so unposed images are dropped here rather than failing later
    in ``resolve_reference_name``. Matching is on the filename stem so that a
    COLMAP model listing ``frame0.png`` still finds ``frame0.jpg`` on disk.

    ``exclude`` drops named images from the database. Its purpose is offline
    testing: replaying a reference image as the query is meaningless if that
    same image is in the database, because retrieval returns it and the solve
    starts from the answer.
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError("Reference image folder not found: {}".format(image_dir))
    names = sorted(
        name for name in os.listdir(image_dir) if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not names:
        raise RuntimeError(
            "No images in {} (looked for {})".format(image_dir, ", ".join(IMAGE_EXTENSIONS))
        )
    if reference_poses:
        posed = _stems(reference_poses)
        names = [name for name in names if os.path.splitext(name)[0] in posed]
    if exclude:
        dropped = _stems(exclude)
        names = [name for name in names if os.path.splitext(name)[0] not in dropped]
    return names


def build_database_from_images(
    image_dir: str,
    reference_poses: Optional[Dict[str, np.ndarray]],
    cache_dir: str,
    force_rebuild: bool = False,
    exclude: Optional[Sequence[str]] = None,
    logger=None,
) -> str:
    """Extract NetVLAD descriptors from the map's raw reference images.

    The h5 keys are the plain file names, which is what the reference poses are
    keyed by, so a retrieval result maps straight onto an initial pose.

    The cache is invalidated by the set of files and their sizes, so adding,
    removing or re-exporting images triggers a rebuild while a plain re-run does
    not. Returns the path to the descriptor file.
    """
    log = logger or (lambda *a: None)
    image_dir = os.path.abspath(image_dir)
    cache_dir = os.path.abspath(cache_dir)
    descriptor_path = os.path.join(cache_dir, DESCRIPTOR_NAME)
    manifest_path = os.path.join(cache_dir, CACHE_MANIFEST)

    posed = list_reference_images(image_dir, reference_poses)
    names = list_reference_images(image_dir, reference_poses, exclude)
    if not names:
        raise RuntimeError(
            "None of the images in {} have a reference pose; check that the "
            "COLMAP model and this folder describe the same capture".format(image_dir)
        )
    if exclude:
        log(
            "Holding %d image(s) out of the retrieval database: %s",
            len(posed) - len(names),
            ", ".join(exclude),
        )
    if reference_poses and len(posed) < len(reference_poses):
        log(
            "%d of %d reference poses have no image file in %s and are not in the "
            "retrieval database",
            len(reference_poses) - len(posed),
            len(reference_poses),
            image_dir,
        )

    fingerprint = hashlib.sha1()
    for name in names:
        fingerprint.update(name.encode("utf-8"))
        fingerprint.update(str(os.path.getsize(os.path.join(image_dir, name))).encode("utf-8"))
    manifest = {
        "source": "images",
        "image_dir": image_dir,
        "num_images": len(names),
        "fingerprint": fingerprint.hexdigest(),
        "descriptor": DESCRIPTOR_NAME,
    }

    if not force_rebuild and _cache_is_current(manifest, descriptor_path, manifest_path, log):
        return descriptor_path

    os.makedirs(cache_dir, exist_ok=True)
    for stale in (descriptor_path, manifest_path):
        if os.path.isfile(stale):
            os.remove(stale)

    log("Building the NetVLAD database from %d raw images in %s", len(names), image_dir)
    descriptor_path = _extract_netvlad(image_dir, names, cache_dir, log)
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    log("NetVLAD database built: %s", descriptor_path)
    return descriptor_path


def build_database_from_map(
    gaussian_map,
    reference_poses: Dict[str, np.ndarray],
    camera,
    cache_dir: str,
    force_rebuild: bool = False,
    render_scale: float = 0.5,
    jpeg_quality: int = 92,
    logger=None,
) -> str:
    """Render every reference pose and extract NetVLAD descriptors from them.

    ``render_scale`` shrinks the database renders. NetVLAD resizes to 1024 px
    internally anyway, so half resolution costs nothing in retrieval quality and
    roughly quarters the rendering time.

    Returns the path to the descriptor file.
    """
    log = logger or (lambda *a: None)
    cache_dir = os.path.abspath(cache_dir)
    image_dir = os.path.join(cache_dir, "db_images")
    descriptor_path = os.path.join(cache_dir, DESCRIPTOR_NAME)
    manifest_path = os.path.join(cache_dir, CACHE_MANIFEST)

    scale = float(render_scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("render_scale must be in (0, 1]")
    db_camera = camera.scaled_to(
        max(1, int(round(camera.width * scale))), max(1, int(round(camera.height * scale)))
    )
    manifest = _manifest_for(
        gaussian_map.model_path,
        gaussian_map.iteration,
        db_camera.width,
        db_camera.height,
        len(reference_poses),
    )

    if not force_rebuild and _cache_is_current(manifest, descriptor_path, manifest_path, log):
        return descriptor_path

    if os.path.isdir(image_dir):
        for name in os.listdir(image_dir):
            os.remove(os.path.join(image_dir, name))
    os.makedirs(image_dir, exist_ok=True)
    for stale in (descriptor_path, manifest_path):
        if os.path.isfile(stale):
            os.remove(stale)

    names = sorted(reference_poses.keys())
    log("Rendering %d reference views at %dx%d for the retrieval database",
        len(names), db_camera.width, db_camera.height)

    written: List[str] = []
    for index, name in enumerate(names):
        rgb, _ = gaussian_map.render(reference_poses[name], db_camera)
        array = (rgb.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        # Flat name so the h5 keys line up with the reference-pose keys.
        out_name = os.path.splitext(os.path.basename(name))[0] + ".jpg"
        if not cv2.imwrite(
            os.path.join(image_dir, out_name), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        ):
            raise IOError("Could not write database render: {}".format(out_name))
        written.append(out_name)
        if (index + 1) % 50 == 0 or index + 1 == len(names):
            log("  rendered %d/%d", index + 1, len(names))

    descriptor_path = _extract_netvlad(image_dir, written, cache_dir, log)
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    log("NetVLAD database built: %s", descriptor_path)
    return descriptor_path


def resolve_database_descriptors(database_path: Optional[str]) -> Optional[str]:
    """Return the prepared descriptor file for a database folder, if present."""
    if not database_path:
        return None
    candidate = os.path.join(os.path.abspath(database_path), "sfm", DESCRIPTOR_NAME)
    return candidate if os.path.isfile(candidate) else None


def _holds_images(path: str) -> bool:
    return os.path.isdir(path) and any(
        name.lower().endswith(IMAGE_EXTENSIONS) for name in os.listdir(path)
    )


def find_reference_image_dir(*roots: str) -> Optional[str]:
    """Locate the raw reference images belonging to a map.

    ``<root>/images`` is the layout COLMAP and 3DGS both use; a root that is
    itself a folder of images is accepted so an explicit path always works.
    """
    for root in roots:
        if not root:
            continue
        root = os.path.abspath(root)
        for candidate in (os.path.join(root, "images"), root):
            if _holds_images(candidate):
                return candidate
    return None
