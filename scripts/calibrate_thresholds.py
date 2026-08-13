"""
For each category, score every real test photo (both normal and defective,
MVTec AD ships both with ground-truth labels) against that category's memory
bank, then pick the cutoff score that best separates "normal" from
"defective" using ROC analysis (Youden's J statistic: maximize
true-positive-rate minus false-positive-rate). Saves the result to
models/thresholds.json, which src/detect.py reads at inference time.

    python -m scripts.calibrate_thresholds
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve

from src.features import extract_patch_features
from src.memory_bank import load_memory_bank, MODELS_DIR

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mvtec_ad"
CATEGORIES = ["screw", "bottle", "leather", "hazelnut"]


def image_score(nn, path: str) -> float:
    img = Image.open(path)
    feats = extract_patch_features(img)
    flat = feats.reshape(-1, feats.shape[-1])
    distances, _ = nn.kneighbors(flat)
    return float(distances.max())


if __name__ == "__main__":
    thresholds = {}
    for category in CATEGORIES:
        nn = load_memory_bank(category)
        test_dir = DATA_DIR / category / "test"

        scores, labels = [], []
        for subfolder in sorted(test_dir.iterdir()):
            is_good = subfolder.name == "good"
            for path in sorted(subfolder.glob("*.png")):
                scores.append(image_score(nn, str(path)))
                labels.append(0 if is_good else 1)

        scores = np.array(scores)
        labels = np.array(labels)
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        best_idx = np.argmax(tpr - fpr)
        best_threshold = float(roc_thresholds[best_idx])

        accuracy = float(((scores >= best_threshold).astype(int) == labels).mean())
        thresholds[category] = best_threshold
        print(f"{category}: threshold={best_threshold:.4f}  test_accuracy={accuracy:.3f}  "
              f"n_good={int((labels == 0).sum())}  n_defective={int((labels == 1).sum())}")

    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    print("Saved models/thresholds.json")
