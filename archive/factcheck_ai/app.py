"""
FactCheck AI – FastAPI main application entry point.
Registers all routers and configures CORS + logging.
"""

import logging
import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend"))
sys.path.append(_backend_root)
sys.path.append(os.path.join(_backend_root, "models/classification"))
sys.path.append(os.path.join(_backend_root, "fact_checker"))
sys.path.append(os.path.join(_backend_root, "screenshot"))

from routers.text_router import router as text_router
from routers.image_router import router as image_router
from routers.screenshot_router import router as screenshot_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FactCheck AI API",
    description="Fake-news detection + AI image manipulation detection backend.",
    version="1.0.0",
)

# Allow the Chrome extension (and local dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your extension origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(text_router,       prefix="/analyze", tags=["Text"])
app.include_router(image_router,      prefix="/analyze", tags=["Image"])
app.include_router(screenshot_router, prefix="/analyze", tags=["Screenshot"])


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
