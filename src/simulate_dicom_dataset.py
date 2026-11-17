"""Simulate 1,220 grayscale medical-image-style scans (64x64) with a
real geometric ground-truth lesion mask (random circular blobs with
noisy intensity) -- standing in for DICOM pixel arrays (see README's
DICOM note), the real input a U-Net segmentation pipeline trains and
evaluates against."""
from __future__ import annotations

import numpy as np

N_IMAGES = 1220
IMG_SIZE = 64
SEED = 967


def _make_image_and_mask(rng: np.random.Generator):
    img = rng.normal(0.3, 0.08, (IMG_SIZE, IMG_SIZE)).clip(0, 1)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    cx, cy = rng.integers(16, 48, 2)
    radius = rng.integers(6, 14)
    yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    dist = (xx - cx) ** 2 + (yy - cy) ** 2
    lesion = dist <= radius ** 2

    img[lesion] += rng.normal(0.35, 0.05)
    img = img.clip(0, 1)
    mask[lesion] = 1.0
    return img.astype(np.float32), mask


def simulate(n_images: int = N_IMAGES, seed: int = SEED):
    rng = np.random.default_rng(seed)
    images = np.zeros((n_images, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    masks = np.zeros((n_images, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i in range(n_images):
        images[i], masks[i] = _make_image_and_mask(rng)
    return images, masks


if __name__ == "__main__":
    images, masks = simulate()
    print(images.shape, masks.mean())
