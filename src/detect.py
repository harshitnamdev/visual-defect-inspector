"""
Inference: score a new photo against a category's memory bank of normal
patches, turn those scores into a heatmap overlaid on the original photo,
and produce an image-level OK/Defective label using a per-category threshold
calibrated in scripts/calibrate_thresholds.py.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from src.features import extract_patch_features
from src.memory_bank import load_memory_bank, MODELS_DIR

THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"

_nn_cache: dict = {}
_thresholds: dict | None = None


def _get_nn(category: str):
    if category not in _nn_cache:
        _nn_cache[category] = load_memory_bank(category)
    return _nn_cache[category]


def _get_threshold(category: str) -> float:
    global _thresholds
    if _thresholds is None:
        _thresholds = json.loads(THRESHOLDS_PATH.read_text())
    return _thresholds[category]


def detect(image: Image.Image, category: str) -> dict:
    feats = extract_patch_features(image)  # [grid_h, grid_w, D]
    grid_h, grid_w, dim = feats.shape
    flat = feats.reshape(-1, dim)

    nn = _get_nn(category)
    distances, _ = nn.kneighbors(flat)  # nearest normal patch, for every patch
    scores = distances.reshape(grid_h, grid_w)

    image_score = float(scores.max())
    threshold = _get_threshold(category)
    label = "Defective" if image_score >= threshold else "OK"

    original = np.array(image.convert("RGB"))
    heatmap = cv2.resize(scores, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original, 0.65, heatmap_color, 0.35, 0)
    overlay_image = Image.fromarray(overlay)

    max_loc = np.unravel_index(np.argmax(scores), scores.shape)
    region_y = "top" if max_loc[0] < grid_h / 3 else ("bottom" if max_loc[0] > 2 * grid_h / 3 else "middle")
    region_x = "left" if max_loc[1] < grid_w / 3 else ("right" if max_loc[1] > 2 * grid_w / 3 else "center")

    return {
        "label": label,
        "score": round(image_score, 4),
        "threshold": round(threshold, 4),
        "region": f"{region_y}-{region_x}",
        "overlay_image": overlay_image,
    }
