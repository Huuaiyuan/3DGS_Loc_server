#!/usr/bin/env python3
"""MASt3R dense matching with an exact pixel mapping back to original images.

``dust3r.utils.image.load_images`` resizes the long edge to 512 and then centre
crops to a multiple of 16, so match coordinates come back in a cropped frame and
have to be mapped to original pixels before they can be used with the camera
intrinsics.

The benchmark scripts re-derived that mapping from the *cropped* size, which
makes the crop offset algebraically collapse to zero: for a 1224x1024 image the
true vertical crop offset is 6 px, and they used 0. They also ignored the
half-pixel centre convention of image resizing, worth another ~0.7 px. Both
errors land directly in the PnP residual, which is thresholded at 1 px.

This module preprocesses images in memory and keeps the exact resize/crop
parameters alongside each view, so the inverse mapping is by construction rather
than re-derivation. Skipping the disk round-trip also drops the JPEG round-trip
the old path put between the renderer and the matcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import PIL.Image
import torch

# Must precede any dust3r import: this repository's top-level ``dust3r/`` is the
# checkout, not the package (which is ``dust3r/dust3r/``). The shim puts the
# checkout on sys.path so ``dust3r.utils`` resolves to the real package.
import mast3r.utils.path_to_dust3r  # noqa: F401

from dust3r.utils.image import ImgNorm


@dataclass
class ViewGeometry:
    """Resize/crop parameters relating a preprocessed view to its original image."""

    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int

    def to_original_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Map (N, 2) cropped-frame coordinates back to original image pixels."""
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.size == 0:
            return pixels.reshape(-1, 2)
        scale_x = float(self.original_width) / float(self.resized_width)
        scale_y = float(self.original_height) / float(self.resized_height)
        out = np.empty_like(pixels)
        # Pixel centres: a resize maps u_src -> (u_src + 0.5) / scale - 0.5.
        out[:, 0] = (pixels[:, 0] + self.crop_left + 0.5) * scale_x - 0.5
        out[:, 1] = (pixels[:, 1] + self.crop_top + 0.5) * scale_y - 0.5
        return out


def preprocess_view(
    rgb_uint8: np.ndarray, index: int, size: int = 512, square_ok: bool = False
) -> Tuple[dict, ViewGeometry]:
    """Build one DUSt3R view dict from an RGB array, mirroring ``load_images``."""
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB array, got {}".format(rgb_uint8.shape))
    image = PIL.Image.fromarray(np.ascontiguousarray(rgb_uint8))
    original_width, original_height = image.size

    # dust3r._resize_pil_image: scale the long edge to `size`.
    long_edge = max(image.size)
    interp = PIL.Image.LANCZOS if long_edge > size else PIL.Image.BICUBIC
    new_size = tuple(int(round(value * size / long_edge)) for value in image.size)
    image = image.resize(new_size, interp)

    resized_width, resized_height = image.size
    cx, cy = resized_width // 2, resized_height // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if not square_ok and resized_width == resized_height:
        halfh = int(3 * halfw / 4)
    image = image.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    geometry = ViewGeometry(
        original_width=original_width,
        original_height=original_height,
        resized_width=resized_width,
        resized_height=resized_height,
        crop_left=cx - halfw,
        crop_top=cy - halfh,
        crop_width=image.size[0],
        crop_height=image.size[1],
    )
    view = dict(
        img=ImgNorm(image)[None],
        true_shape=np.int32([image.size[::-1]]),
        idx=index,
        instance=str(index),
    )
    return view, geometry


def tensor_to_rgb_uint8(rgb: torch.Tensor) -> np.ndarray:
    """Convert a CHW float tensor in [0, 1] to an HxWx3 uint8 RGB array."""
    array = rgb.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (array * 255.0).round().astype(np.uint8)


@torch.no_grad()
def match_pair(
    model,
    rgb_a: np.ndarray,
    rgb_b: np.ndarray,
    device: str = "cuda",
    subsample: int = 8,
    border: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match two RGB images, returning correspondences in original pixels.

    Returns ``(pixels_a, pixels_b)``, both (N, 2) float64 arrays.
    """
    from dust3r.inference import inference
    from mast3r.fast_nn import fast_reciprocal_NNs

    view_a, geometry_a = preprocess_view(rgb_a, 0)
    view_b, geometry_b = preprocess_view(rgb_b, 1)

    output = inference([(view_a, view_b)], model, device, batch_size=1, verbose=False)
    pred1, pred2 = output["pred1"], output["pred2"]
    desc_a = pred1["desc"].squeeze(0).detach()
    desc_b = pred2["desc"].squeeze(0).detach()

    matches_a, matches_b = fast_reciprocal_NNs(
        desc_a,
        desc_b,
        subsample_or_initxy1=subsample,
        device=device,
        dist="dot",
        block_size=2 ** 13,
    )
    matches_a = np.asarray(matches_a).reshape(-1, 2)
    matches_b = np.asarray(matches_b).reshape(-1, 2)
    if matches_a.shape[0] == 0:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty

    height_a, width_a = geometry_a.crop_height, geometry_a.crop_width
    height_b, width_b = geometry_b.crop_height, geometry_b.crop_width
    valid = (
        (matches_a[:, 0] >= border)
        & (matches_a[:, 0] < width_a - border)
        & (matches_a[:, 1] >= border)
        & (matches_a[:, 1] < height_a - border)
        & (matches_b[:, 0] >= border)
        & (matches_b[:, 0] < width_b - border)
        & (matches_b[:, 1] >= border)
        & (matches_b[:, 1] < height_b - border)
    )
    matches_a = matches_a[valid]
    matches_b = matches_b[valid]

    return (
        geometry_a.to_original_pixels(matches_a),
        geometry_b.to_original_pixels(matches_b),
    )
