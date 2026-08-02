"""
routers/text_router.py
-----------------------
POST /analyze/text
"""

import logging
import os
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
try:
    from checker_ar import run
except ImportError:
    def run(text: str):
        return {
            "claim": text,
            "final_prediction": "SUPPORTS",
            "confidence": 0.95,
            "evidence": [],
            "explanation": "Archived prototype fallback response."
        }
logger = logging.getLogger(__name__)
router = APIRouter()

class TextRequest(BaseModel):
    text: str

from typing import Any

class TextResponse(BaseModel):
    message: str
    result: Any

MOCK=False
SAVE_DIR = "saved_texts"
os.makedirs(SAVE_DIR, exist_ok=True)

#dirrectory to save jsons
JSON_DIR = "saved_json"
os.makedirs(JSON_DIR, exist_ok=True)


@router.post("/text", summary="Fact-check text for fake news")
async def analyze_text_endpoint(payload: TextRequest):
    text = payload.text

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text value is required")

    try:
        if MOCK:
            return {
            "claim": text,
            "final_prediction": "SUPPORTS",
            "confidence": 0.95,
            "evidence": ["www.example1.com","www.example2.com","www.example3.com","www.example4.com","www.example5.com"],
            "explanation": "This is a mock response."
        }
        result = run(text)
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        name= f"{date}_text"
        #save result as json file in saved_json folder
        with open(os.path.join(JSON_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        return result
    except Exception as e:
        logger.exception("Failed to analyze text")
        raise HTTPException(status_code=500, detail=str(e))
