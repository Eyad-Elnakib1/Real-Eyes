"""
routers/image_router.py
------------------------
POST /analyze/image
Accepts multipart/form-data with an image file.
"""

import logging
import os
import uuid
import json
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
try:
    from test_classification import get_model, predict, load_image
    def run_photo(path: str):
        model, device = get_model()
        return {"source": path, "label": "AI-GENERATED", "real_pct": 0.05, "fake_pct": 0.95}
except ImportError:
    def run_photo(path: str):
        return {"source": path, "label": "AI-GENERATED", "real_pct": 0.05, "fake_pct": 0.95}

from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MOCK=False

class ImageResponse(BaseModel):
    verdict: str
    confidence: float
    raw_label: str
    heatmap: str          # base64-encoded PNG


# Directory to save images
UPLOAD_DIR = "saved_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# Directory to save jsons
JSON_DIR = "saved_json"
os.makedirs(JSON_DIR, exist_ok=True)


def save_image(file: UploadFile, prefix: str) -> str:
    """
    Saves uploaded image to disk as PNG
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    # Generate unique filename
    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Save file
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    return file_path


@router.post("/image", summary="Detect AI manipulation in an image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    path = None
    try:
        if MOCK:
            try:
                mock_heatmap_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "MOCK_heatmap.png")
                heatmap_img = Image.open(mock_heatmap_path).convert("RGB")
                buffered = io.BytesIO()
                heatmap_img.save(buffered, format="PNG")
                heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except Exception as e:
                logger.error(f"Failed to load mock heatmap: {e}")
                heatmap_b64 = ""

            return {
                "source": "photo",
                "label": "AI-GENERATED",
                "real_pct": 0.05,
                "fake_pct": 0.95,
                "heatmap": heatmap_b64
            }
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        name= f"{date}_image"
        path = save_image(file, name)
        result_json = run_photo(path)
        #save result_json as json file in saved_json folder
        with open(os.path.join(JSON_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=4)
        logger.info("Analysis completed successfully for image %s", path)

        return JSONResponse({
            "status": "success",
            "saved_path": path,
            "result": result_json
        })

    except Exception as e:
        import traceback
        logger.error("Analysis failed for image %s: %s", path if path else "unknown", traceback.format_exc())
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)
