"""
One-time offline step: for each supported category, build its memory bank
from the "good" training photos.

    python -m scripts.build_memory_banks
"""
from pathlib import Path
from src.memory_bank import build_memory_bank

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mvtec_ad"
CATEGORIES = ["screw", "bottle", "leather", "hazelnut"]

if __name__ == "__main__":
    for category in CATEGORIES:
        good_dir = DATA_DIR / category / "train" / "good"
        image_paths = sorted(str(p) for p in good_dir.glob("*.png"))
        print(f"{category}: building memory bank from {len(image_paths)} good images...")
        build_memory_bank(category, image_paths)
        print(f"{category}: done, saved to models/{category}_memory_bank.npy")
