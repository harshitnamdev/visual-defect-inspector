"""
Builds and loads the per-category "memory bank" of normal patch appearances.

Building: extract patch features from every known-good training photo of a
category, pool every patch from every photo together, then compress that
pool down to a few thousand representative points via k-means -- keeping
just the cluster centers. This is what makes the memory bank small and
lookups fast, without losing meaningful diversity in what "normal" covers.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from src.features import extract_patch_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
N_CLUSTERS = 2000


def build_memory_bank(category: str, good_image_paths: list[str]) -> None:
    all_patches = []
    for i, path in enumerate(good_image_paths):
        img = Image.open(path)
        feats = extract_patch_features(img)  # [H, W, D]
        all_patches.append(feats.reshape(-1, feats.shape[-1]))
        if (i + 1) % 25 == 0 or (i + 1) == len(good_image_paths):
            print(f"  [{category}] extracted features from {i + 1}/{len(good_image_paths)} images", flush=True)
    all_patches = np.concatenate(all_patches, axis=0)  # [N_total_patches, D]

    n_clusters = min(N_CLUSTERS, len(all_patches))
    print(f"  [{category}] clustering {len(all_patches)} patches into {n_clusters} centers...", flush=True)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, n_init=1, max_iter=100, random_state=42)
    kmeans.fit(all_patches)
    memory_bank = kmeans.cluster_centers_.astype(np.float32)  # [n_clusters, D]

    MODELS_DIR.mkdir(exist_ok=True)
    np.save(MODELS_DIR / f"{category}_memory_bank.npy", memory_bank)


def load_memory_bank(category: str) -> NearestNeighbors:
    memory_bank = np.load(MODELS_DIR / f"{category}_memory_bank.npy")
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(memory_bank)
    return nn
