"""
routers/screenshot_router.py
-----------------------------
POST /analyze/screenshot
Accepts multipart/form-data with a screenshot image.
"""

import logging
import os
import uuid
import json
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
try:
    from handler import run_extraction as run_screenshot
except ImportError:
    def run_screenshot(path: str):
        return {"extracted_text": "Mock OCR Text", "image_analysis": []}

try:
    from checker_ar import run
except ImportError:
    def run(text: str):
        return {"verdict": "Fake", "confidence": 0.88}


from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILE_SIZE = 30 * 1024 * 1024  # 30 MB
MOCK = False

class ScreenshotResponse(BaseModel):
    extracted_text: str
    fake_news_result: Optional[dict] = None
    image_analysis: Optional[dict] = None


# Directory to save images
UPLOAD_DIR = "saved_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
#dirrectory to save jsons
JSON_DIR ="saved_json"
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


@router.post(
    "/screenshot",
    summary="Analyse a screenshot (OCR + fake news + AI image detection)",
)
async def analyze_screenshot_endpoint(file: UploadFile = File(...)):
    path = None
    try:
        if MOCK:
            # Mock results generated for demo purposes
            try:
                mock_heatmap_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "MOCK_heatmap.png")
                heatmap_img = Image.open(mock_heatmap_path).convert("RGB")
                buffered = io.BytesIO()
                heatmap_img.save(buffered, format="PNG")
                heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except Exception as e:
                logger.error(f"Failed to load mock heatmap: {e}")
                heatmap_b64 = ""

            results = {
                "extracted_text": "Breaking News: The new policy will immediately affect all citizens starting next week. Experts warn of severe economic consequences.\n\nLocal authorities have not yet commented on the situation.",
                "fake_news_result": {
                    "verdict": "Fake",
                    "confidence": 0.88
                },
                "image_analysis": [{
                    "verdict": "AI Modified",
                    "confidence": 0.94,
                    "heatmap": heatmap_b64
                },
                {
                    "verdict": "Real",
                    "confidence": 0.86,
                    "heatmap": heatmap_b64
                }]
            }
            return results
        # Check if file is an image
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File is not an image")
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        name= f"{date}_screenshot"
        path = save_image(file, name)
        result_json = run_screenshot(path)


        #run text verifier on extracted text
        text_result = run(result_json["extracted_text"])
        


        result_json["fake_news_result"] = text_result
        #save result_json as json file in saved_json folder
        with open(os.path.join(JSON_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=4)
        logger.info("Analysis completed successfully for screenshot %s", path)

        return result_json


    except Exception as e:
        import traceback
        logger.error("Analysis failed for screenshot %s: %s", path if path else "unknown", traceback.format_exc())
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)
