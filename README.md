# Visual Defect Inspector

A computer-vision quality-control tool: upload a photo of a product and it tells you whether it's defective, highlights exactly *where* the defect is on the photo, and writes a one-sentence report — no manual visual inspection needed.

**Live demo:** [App](https://visual-defect-inspector.vercel.app) · [API docs](https://visual-defect-inspector-2u6j.onrender.com/docs)

## How it works

This is **not** a trained classifier — it's a PatchCore-style anomaly detector that only ever learns what a *normal* product looks like:

1. **Feature extraction** — a frozen, ImageNet-pretrained ResNet18 (never fine-tuned) turns every photo into a grid of patch-level feature vectors, combining mid-level texture detail and higher-level shape information.
2. **Memory bank** — for each product category, patches from hundreds of known-good photos are pooled and compressed via k-means into ~2000 representative "normal" reference points.
3. **Detection** — a new photo's patches are compared against that memory bank via nearest-neighbor distance. Patches far from anything in the memory bank are flagged as anomalous, rendered as a heatmap directly on the original photo.
4. **Narration** — an LLM turns the already-computed result into one plain sentence. It never decides whether something is defective; that's fully deterministic. If the LLM is unavailable or its phrasing doesn't match the real result, a template sentence is used instead — the app is always correct, the LLM is only polish.

## Tech stack

| Layer | Tech |
|---|---|
| Feature extraction | PyTorch, torchvision (ResNet18, frozen) |
| Anomaly detection | scikit-learn (MiniBatchKMeans, NearestNeighbors) |
| Backend | FastAPI |
| LLM narration | Qwen 3.6 35B (via [OpenRouter](https://openrouter.ai)) |
| Frontend | React + Vite |
| Backend hosting | Render (Docker) |
| Frontend hosting | Vercel |

## Dataset

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — a real industrial visual-inspection benchmark. 4 categories are supported: screw, bottle, leather, hazelnut, each with hundreds of defect-free training photos and labeled test photos (both normal and defective, with pixel-level ground truth).

Per-category detection accuracy on real MVTec AD test photos (ROC-calibrated threshold): bottle 96.4%, leather 93.5%, hazelnut 90.9%, screw 71.9% (screws have small, subtle defects and are a known harder category in this benchmark).

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/detect` | POST | multipart form: `category`, `image` → `{ category, label, score, threshold, region, narrative, overlay_image_base64 }` |

## Running locally

### Backend
```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash; use venv\Scripts\activate on cmd
pip install -r requirements.txt

# create .env with:
# OPENROUTER_API_KEY=your_key_here

# one-time: build memory banks + calibrate thresholds from the MVTec AD dataset
python -m scripts.build_memory_banks
python -m scripts.calibrate_thresholds

uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Set `VITE_API_URL` in `frontend/.env` to point at your running backend.

## Project structure
```
├── app/            # FastAPI application (routes)
├── src/            # Feature extraction, memory bank, detection, LLM narration
├── scripts/        # Offline: build memory banks, calibrate thresholds, test pipeline
├── models/         # Saved memory banks + calibrated thresholds (small, committed)
├── frontend/       # React + Vite upload/results UI
├── Dockerfile      # Render deployment
└── Procfile        # Alternate Railway-style start command
```

## License

MIT
