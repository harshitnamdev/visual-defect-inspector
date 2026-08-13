import base64
import io
import logging
import traceback

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from src.detect import detect
from src.narrate import narrate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visual-defect-inspector")

app = FastAPI(title="Visual Defect Inspector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://visual-defect-inspector.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_CATEGORIES = ["screw", "bottle", "leather", "hazelnut"]


class DetectResponse(BaseModel):
    category: str
    label: str
    score: float
    threshold: float
    region: str
    narrative: str
    overlay_image_base64: str


@app.get("/")
def root():
    return {"status": "ok", "service": "visual-defect-inspector", "categories": SUPPORTED_CATEGORIES}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/detect", response_model=DetectResponse)
async def run_detect(category: str = Form(...), image: UploadFile = File(...)):
    if category not in SUPPORTED_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"category must be one of {SUPPORTED_CATEGORIES}")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))

        result = detect(pil_image, category)
        narrative = narrate(category, result)

        buf = io.BytesIO()
        result["overlay_image"].save(buf, format="PNG")
        overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return DetectResponse(
            category=category,
            label=result["label"],
            score=result["score"],
            threshold=result["threshold"],
            region=result["region"],
            narrative=narrative,
            overlay_image_base64=overlay_b64,
        )
    except Exception:
        logger.error("detect endpoint failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Detection failed. Please try a different image.")
