"""
Screenshot Text & Image Extractor
===================================
Extracts text and visual regions from screenshots.
"""

from PIL import Image, ImageDraw
import json
import os
from pathlib import Path
import cv2
import numpy as np
import easyocr
import arabic_reshaper
from bidi.algorithm import get_display


photo = "pho2.png"


# ── OCR & Detection ───────────────────────────────────────────────────────────

def is_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return any('؀' <= c <= 'ۿ' for c in text)


def reconstruct_lines(blocks: list) -> str:
    """
    Group text blocks by their vertical position (same line),
    then sort each line RTL if Arabic, LTR if English.
    Returns clean multi-line string.
    """
    if not blocks:
        return ""

    # Sort all blocks top-to-bottom first
    sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][0][1])

    # Group into lines — blocks whose vertical center is within LINE_THRESHOLD px
    LINE_THRESHOLD = 15
    lines = []
    current_line = [sorted_blocks[0]]

    for block in sorted_blocks[1:]:
        cy_current = sum(p[1] for p in block["bbox"]) / 4
        cy_prev    = sum(p[1] for p in current_line[-1]["bbox"]) / 4
        if abs(cy_current - cy_prev) <= LINE_THRESHOLD:
            current_line.append(block)
        else:
            lines.append(current_line)
            current_line = [block]
    lines.append(current_line)

    result_lines = []
    for line in lines:
        # Detect direction by majority of blocks in this line
        arabic_count = sum(1 for b in line if is_arabic(b["text"]))
        is_rtl = arabic_count > len(line) / 2

        if is_rtl:
            # Sort RTL: rightmost block first (highest x = first word)
            line_sorted = sorted(line, key=lambda b: b["bbox"][0][0], reverse=True)
        else:
            # Sort LTR: leftmost block first
            line_sorted = sorted(line, key=lambda b: b["bbox"][0][0])

        #result_lines.append(" ".join(fix_arabic(b["text"]) for b in line_sorted))
        result_lines.append(" ".join(b["text"] for b in line_sorted))
        
        

    return "\n".join(result_lines)


def extract_text(image_path: str) -> list:
    """Extract text with bounding boxes using EasyOCR."""
    reader = easyocr.Reader(['en', 'ar'], gpu=False)
    results = reader.readtext(image_path)
    extracted = []
    for (bbox, text, confidence) in results:
        if confidence > 0.3:
            extracted.append({
                "text": text,
                "confidence": round(confidence, 2),
                "bbox": [[int(p[0]), int(p[1])] for p in bbox]
            })
    return extracted


def detect_image_regions(image_path: str, text_results: list = None) -> list:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # ── Step 1: mask text areas ───────────────────────────────────────────────
    text_mask = np.zeros((h, w), dtype=np.uint8)
    if text_results:
        padding = max(10, min(w, h) // 50)
        for item in text_results:
            pts = np.array(item["bbox"], dtype=np.int32)
            x1 = max(0, pts[:, 0].min() - padding)
            y1 = max(0, pts[:, 1].min() - padding)
            x2 = min(w, pts[:, 0].max() + padding)
            y2 = min(h, pts[:, 1].max() + padding)
            text_mask[y1:y2, x1:x2] = 255

    # ── Step 2: detect solid-color UI bars (headers/toolbars) ─────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_std = gray.std(axis=1)
    UI_STD_THRESHOLD = 18

    ui_mask = np.zeros((h, w), dtype=np.uint8)
    in_band = False
    band_start = 0
    for row_idx, std in enumerate(row_std):
        if std < UI_STD_THRESHOLD and not in_band:
            in_band = True
            band_start = row_idx
        elif std >= UI_STD_THRESHOLD and in_band:
            in_band = False
            band_h = row_idx - band_start
            if band_h >= max(4, int(h * 0.015)):
                ui_mask[band_start:row_idx, :] = 255
    if in_band:
        band_h = h - band_start
        if band_h >= max(4, int(h * 0.015)):
            ui_mask[band_start:h, :] = 255

    combined_mask = cv2.bitwise_or(text_mask, ui_mask)

    # ── Step 3: variance map → photo pixels ───────────────────────────────────
    blurred      = cv2.GaussianBlur(img, (21, 21), 0).astype(np.float32)
    diff         = (img.astype(np.float32) - blurred) ** 2
    variance_map = diff.mean(axis=2)

    photo_mask = (variance_map > 8).astype(np.uint8) * 255
    photo_mask = cv2.bitwise_and(photo_mask, cv2.bitwise_not(combined_mask))

    # ── Step 4: morphological close/open ─────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    closed = cv2.morphologyEx(photo_mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed,     cv2.MORPH_OPEN,  kernel)

    # ── Step 5: find contours ─────────────────────────────────────────────────
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (w * h) * 0.02
    max_area = (w * h) * 0.95

    regions = []
    seen    = set()
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        if area < min_area or area > max_area:
            continue

        region_center_y = y + rh // 2
        if ui_mask[region_center_y, w // 2] == 255:
            continue

        key = (x // 30, y // 30, rw // 30, rh // 30)
        if key in seen:
            continue
        seen.add(key)

        # ── NEW: expand the bounding box upward into low-variance photo content ──
        # Walk upward row-by-row from the detected top edge. Keep expanding as long
        # as the row is NOT a UI bar and has reasonable image content (not pure solid
        # background of the app, which tends to be very dark or very uniform).
        expanded_y = y
        for probe_y in range(y - 1, -1, -1):
            if ui_mask[probe_y, x + rw // 2] == 255:
                break  # hit a UI bar — stop

            row_slice = img[probe_y, x:x + rw]
            # Check if this row looks like app background:
            # App backgrounds are typically very dark (mean < 30) OR extremely uniform (std < 6)
            row_gray  = gray[probe_y, x:x + rw]
            row_mean  = float(row_gray.mean())
            row_s     = float(row_gray.std())

            if row_mean < 30 and row_s < 6:
                break  # pure dark app background — stop expanding
            if row_s < 4:
                break  # completely flat row — likely not photo content

            expanded_y = probe_y  # this row looks like it belongs to the photo

        # Also try expanding downward (same logic, catches bottom clips)
        expanded_y2 = y + rh
        for probe_y in range(y + rh, h):
            if ui_mask[probe_y, x + rw // 2] == 255:
                break
            row_gray = gray[probe_y, x:x + rw]
            row_mean = float(row_gray.mean())
            row_s    = float(row_gray.std())
            if row_mean < 30 and row_s < 6:
                break
            if row_s < 4:
                break
            expanded_y2 = probe_y + 1

        new_y  = expanded_y
        new_rh = expanded_y2 - expanded_y

        aspect = rw / new_rh if new_rh > 0 else 1
        if rw > w * 0.25 and new_rh > h * 0.25:
            region_type = "photo / main image"
        elif aspect > 3:
            region_type = "banner / horizontal bar"
        elif aspect < 0.4:
            region_type = "vertical image"
        else:
            region_type = "image block"

        regions.append({
            "id":      len(regions) + 1,
            "type":    region_type,
            "x": x,        "y": new_y,
            "width":   rw, "height": new_rh,
            "area_px": rw * new_rh,
        })

    regions.sort(key=lambda r: r["area_px"], reverse=True)
    return regions[:10]


def annotate_image(image_path: str, text_results: list, regions: list) -> Image.Image:
    """Draw bounding boxes: green for text, orange for image regions."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    for region in regions:
        x, y, rw, rh = region["x"], region["y"], region["width"], region["height"]
        draw.rectangle([x, y, x + rw, y + rh], outline=(255, 140, 0, 220), width=2)

    for item in text_results:
        pts = item["bbox"]
        flat = [coord for pt in pts for coord in pt]
        draw.polygon(flat, outline=(0, 220, 100, 230))

    return img


def classify_text(text_blocks: list, regions: list) -> tuple:
    """
    Split text_blocks into two lists:
    - outside_text: text that appears outside all image regions (UI / captions)
    - inside_text:  text that appears inside an image region (embedded in photo)
    """
    outside, inside = [], []
    for item in text_blocks:
        pts = item["bbox"]
        cx  = sum(p[0] for p in pts) / 4   # center x of text box
        cy  = sum(p[1] for p in pts) / 4   # center y of text box
        in_region = False
        for r in regions:
            if (r["x"] <= cx <= r["x"] + r["width"] and
                    r["y"] <= cy <= r["y"] + r["height"]):
                in_region = True
                break
        (inside if in_region else outside).append(item)
    return outside, inside


def run_extraction(image_path: str) -> dict:
    """Run full pipeline and return results dict."""
    text_results = extract_text(image_path)
    regions      = detect_image_regions(image_path, text_results)
    outside_text, inside_text = classify_text(text_results, regions)

    cropped_path = None
    if regions:
        cropped_path = save_cropped_region(image_path, regions[0])

    return {
        "source":       image_path,
        "text":         reconstruct_lines(outside_text),
        "text_inside":  reconstruct_lines(inside_text),
        "text_blocks":  text_results,
        "outside_text": outside_text,
        "inside_text":  inside_text,
        "image_regions":  regions,
        "cropped_image":  cropped_path,
        "summary": {
            "total_text_blocks":        len(text_results),
            "total_outside_text":       len(outside_text),
            "total_inside_text":        len(inside_text),
            "total_image_regions":      len(regions),
        }
    }


def save_cropped_region(image_path: str, region: dict) -> str:
    """Save the first detected image region as a PNG and return its path."""
    img = cv2.imread(image_path)
    x, y, w, h = region["x"], region["y"], region["width"], region["height"]
    cropped = img[y:y+h, x:x+w]
    out_path = str(Path(image_path).with_suffix("")) + "_cropped_region.png"
    cv2.imwrite(out_path, cropped)
    return out_path


def save_results(result: dict, image_path: str) -> str:
    base = Path(image_path).with_suffix("")
    out_path = str(base) + "_extraction.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return out_path

def fix_arabic(text: str) -> str:
    """Reshape Arabic text so letters connect properly."""
    if is_arabic(text):
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    return text

# ── Entry point ───────────────────────────────────────────────────────────────

