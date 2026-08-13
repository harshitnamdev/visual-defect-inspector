"""
Sanity-check the full pipeline against real MVTec AD test photos: for each
category, run one known-good and one known-defective photo through
detect() + narrate() and print the result.

    python -m scripts.test_detect
"""
from pathlib import Path
from PIL import Image
from src.detect import detect
from src.narrate import narrate

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mvtec_ad"
CATEGORIES = ["screw", "bottle", "leather", "hazelnut"]

if __name__ == "__main__":
    for category in CATEGORIES:
        test_dir = DATA_DIR / category / "test"
        good_path = next((test_dir / "good").glob("*.png"))
        defect_dirs = [d for d in test_dir.iterdir() if d.name != "good"]
        defect_path = next(defect_dirs[0].glob("*.png"))

        for label, path in [("GOOD photo", good_path), ("DEFECTIVE photo", defect_path)]:
            img = Image.open(path)
            result = detect(img, category)
            narrative = narrate(category, result)
            print("=" * 70)
            print(f"{category} -- {label} ({path.name})")
            print(f"  label={result['label']}  score={result['score']}  threshold={result['threshold']}  region={result['region']}")
            print(f"  narrative: {narrative}")
