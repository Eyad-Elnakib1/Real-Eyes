import io
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

import timm
from torchvision import transforms




from pathlib import Path


# =============================================================================
# CONFIG
# =============================================================================
class CFG:
    IMG_SIZE    = 384
    MODEL_NAME  = "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k"
    NUM_CLASSES = 2
    DROP_PATH   = 0.2

BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "bestv_3.3.pth"
OUT_DIR     = "results"
USE_EMA     = True
CLASS_NAMES = ["REAL", "FAKE"]
VALID_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =============================================================================
# MODEL
# =============================================================================
class FFTBranch(nn.Module):
    def __init__(self, out_dim=256, img_size=384):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, out_dim), nn.GELU(),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x_norm):
        x = x_norm * self.std + self.mean
        fft = torch.fft.fft2(x, norm="ortho")
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.log1p(fft.abs())
        m = mag.mean(dim=(-2, -1), keepdim=True)
        s = mag.std (dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return self.cnn((mag - m) / s)


class AuxBranch(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, out_dim), nn.GELU(),
        )

    def forward(self, aux):
        return self.cnn(aux)


class SwinForensicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG.MODEL_NAME, pretrained=False, num_classes=0,
            global_pool="avg", drop_path_rate=CFG.DROP_PATH, img_size=CFG.IMG_SIZE,
        )
        feat_dim = self.backbone.num_features
        self.fft_branch = FFTBranch(256, CFG.IMG_SIZE)
        self.aux_branch = AuxBranch(256)
        fused_dim = feat_dim + 256 + 256
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, CFG.NUM_CLASSES),
        )

    def forward(self, img_rgb, aux):
        f_swin = self.backbone(img_rgb)
        f_fft  = self.fft_branch(img_rgb)
        f_aux  = self.aux_branch(aux)
        return self.head(torch.cat([f_swin, f_fft, f_aux], dim=1))


# =============================================================================
# PREPROCESS
# =============================================================================
val_tf = transforms.Compose([
    transforms.Resize(int(CFG.IMG_SIZE * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(CFG.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def build_aux(img_rgb_norm):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_uint = (img_rgb_norm * std + mean).clamp(0, 1)
    rgb_np = (rgb_uint.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    noise = (gray.astype(np.float32) - blurred.astype(np.float32)) / 32.0
    noise_t = torch.from_numpy(noise).unsqueeze(0).float()

    v = np.median(gray)
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lo, hi).astype(np.float32) / 255.0
    edges_t = torch.from_numpy(edges).unsqueeze(0).float()
    return torch.cat([noise_t, edges_t], dim=0)


def load_image(path):
    with open(path, "rb") as f:
        img = Image.open(io.BytesIO(f.read()))
        img.load()
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


# =============================================================================
# FILE DISCOVERY
# =============================================================================
def gather_inputs(path):
    if any(ch in path for ch in "*?[]"):
        files = sorted(glob.glob(path))
    elif os.path.isdir(path):
        files = [os.path.join(path, fn) for fn in sorted(os.listdir(path))
                 if os.path.splitext(fn)[1].lower() in VALID_EXTS]
    elif os.path.isfile(path):
        files = [path]
    else:
        files = []
    return [f for f in files if os.path.splitext(f)[1].lower() in VALID_EXTS]


# =============================================================================
# FONTS
# =============================================================================
def load_font(size, bold=True):
    candidates_bold = ["arialbd.ttf", "DejaVuSans-Bold.ttf",
                       "C:/Windows/Fonts/arialbd.ttf",
                       "C:/Windows/Fonts/seguisb.ttf"]
    candidates_reg  = ["arial.ttf", "DejaVuSans.ttf",
                       "C:/Windows/Fonts/arial.ttf",
                       "C:/Windows/Fonts/segoeui.ttf"]
    for fp in (candidates_bold if bold else candidates_reg):
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def text_size(draw, text, font):
    try:
        b = draw.textbbox((0, 0), text, font=font)
        return b[2] - b[0], b[3] - b[1]
    except AttributeError:
        return draw.textsize(text, font=font)


# =============================================================================
# REPORT-STYLE OUTPUT
# =============================================================================
def make_report(pil_img, probs, pred, save_path):
    """
    Create a polished report card:
      [ photo with subtle dark vignette + verdict badge | side panel with stats ]
    """
    real_p = float(probs[0]) * 100
    fake_p = float(probs[1]) * 100
    confidence = max(real_p, fake_p)

    # ----- Layout sizes -----
    photo_h = 720
    src = pil_img.copy()
    sw, sh = src.size
    scale  = photo_h / sh
    photo_w = int(sw * scale)
    photo   = src.resize((photo_w, photo_h), Image.LANCZOS)

    panel_w = 520
    pad     = 36
    canvas_w = photo_w + panel_w
    canvas_h = photo_h

    # ----- Theme -----
    if pred == 0:  # REAL  -> emerald
        accent      = (52, 211, 153)
        accent_dark = (16, 122, 87)
        verdict_txt = "AUTHENTIC"
        sub_txt     = "Likely a real photograph"
    else:           # FAKE -> rose/red
        accent      = (244, 94, 94)
        accent_dark = (153, 27, 27)
        verdict_txt = "AI-GENERATED"
        sub_txt     = "Likely synthetic / manipulated"

    bg_dark   = (17, 22, 31)
    bg_panel  = (24, 30, 41)
    text_main = (236, 240, 247)
    text_mut  = (148, 158, 175)
    track     = (40, 48, 62)

    # ----- Canvas -----
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_dark)

    # ----- Left: photo with bottom-gradient vignette -----
    canvas.paste(photo, (0, 0))
    overlay = Image.new("RGBA", (photo_w, photo_h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    grad_h = int(photo_h * 0.42)
    for i in range(grad_h):
        a = int(220 * (i / grad_h) ** 1.6)
        od.line([(0, photo_h - grad_h + i), (photo_w, photo_h - grad_h + i)],
                fill=(0, 0, 0, a))
    canvas.paste(overlay, (0, 0), overlay)

    draw = ImageDraw.Draw(canvas, "RGBA")

    # ----- Verdict badge (bottom-left of photo) -----
    f_badge = load_font(34, bold=True)
    f_sub   = load_font(18, bold=False)
    bw, bh  = text_size(draw, verdict_txt, f_badge)
    sw_, sh_ = text_size(draw, sub_txt, f_sub)

    badge_pad_x, badge_pad_y = 22, 14
    badge_w = max(bw, sw_) + badge_pad_x * 2
    badge_h = bh + sh_ + badge_pad_y * 2 + 8
    bx = pad
    by = photo_h - badge_h - pad

    # accent vertical bar + dark translucent card
    draw.rounded_rectangle([(bx, by), (bx + badge_w, by + badge_h)],
                           radius=14, fill=(0, 0, 0, 170))
    draw.rounded_rectangle([(bx, by), (bx + 6, by + badge_h)],
                           radius=3, fill=accent + (255,))
    draw.text((bx + badge_pad_x, by + badge_pad_y), verdict_txt,
              font=f_badge, fill=accent + (255,))
    draw.text((bx + badge_pad_x, by + badge_pad_y + bh + 6), sub_txt,
              font=f_sub, fill=text_main + (220,))

    # ----- Right side panel -----
    px = photo_w
    draw.rectangle([(px, 0), (canvas_w, canvas_h)], fill=bg_panel)
    # accent strip
    draw.rectangle([(px, 0), (px + 4, canvas_h)], fill=accent)

    f_title = load_font(20, bold=True)
    f_h1    = load_font(28, bold=True)
    f_pct   = load_font(22, bold=True)
    f_lbl   = load_font(17, bold=False)
    f_small = load_font(14, bold=False)

    cx = px + pad
    cy = pad + 8

    # Header
    draw.text((cx, cy), "DETECTION REPORT", font=f_title, fill=text_mut)
    cy += 30
    draw.text((cx, cy), "Image Authenticity", font=f_h1, fill=text_main)
    cy += 48

    # Confidence ring
    ring_size = 180
    ring_x = cx
    ring_y = cy
    # draw ring on a transparent RGBA layer for smooth arcs
    ring_layer = Image.new("RGBA", (ring_size, ring_size), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring_layer)
    ring_w = 16
    rd.ellipse([(0, 0), (ring_size, ring_size)],
               outline=track + (255,), width=ring_w)
    sweep = 360 * (confidence / 100.0)
    rd.arc([(0, 0), (ring_size, ring_size)],
           start=-90, end=-90 + sweep, fill=accent + (255,), width=ring_w)
    canvas.paste(ring_layer, (ring_x, ring_y), ring_layer)

    # Ring text (centered)
    conf_txt = f"{confidence:.1f}%"
    f_conf   = load_font(34, bold=True)
    cw, ch = text_size(draw, conf_txt, f_conf)
    draw.text((ring_x + (ring_size - cw) // 2,
               ring_y + (ring_size - ch) // 2 - 8),
              conf_txt, font=f_conf, fill=text_main)
    lbl = "CONFIDENCE"
    lw, lh = text_size(draw, lbl, f_small)
    draw.text((ring_x + (ring_size - lw) // 2,
               ring_y + (ring_size + ch) // 2 + 2),
              lbl, font=f_small, fill=text_mut)

    # Right of ring: verdict tag
    tag_x = ring_x + ring_size + 24
    tag_y = ring_y + 14
    draw.text((tag_x, tag_y), "VERDICT", font=f_small, fill=text_mut)
    draw.text((tag_x, tag_y + 22), CLASS_NAMES[pred],
              font=load_font(40, bold=True), fill=accent)
    draw.text((tag_x, tag_y + 76), "MODEL", font=f_small, fill=text_mut)
    draw.text((tag_x, tag_y + 96), "Swin-V2 Forensic",
              font=load_font(18, bold=True), fill=text_main)

    cy = ring_y + ring_size + 40

    # ----- Probability bars -----
    def draw_stat(y, label, pct, color):
        draw.text((cx, y), label, font=f_lbl, fill=text_main)
        pct_txt = f"{pct:.2f}%"
        pw, ph = text_size(draw, pct_txt, f_pct)
        draw.text((canvas_w - pad - pw, y - 3), pct_txt, font=f_pct, fill=color)
        # track
        bar_y = y + 30
        bar_w = canvas_w - cx - pad
        bar_h = 14
        draw.rounded_rectangle([(cx, bar_y), (cx + bar_w, bar_y + bar_h)],
                               radius=7, fill=track)
        fill_w = int(bar_w * pct / 100.0)
        if fill_w > 0:
            draw.rounded_rectangle([(cx, bar_y), (cx + fill_w, bar_y + bar_h)],
                                   radius=7, fill=color)
        return bar_y + bar_h + 28

    cy = draw_stat(cy, "Real",        real_p, (52, 211, 153))
    cy = draw_stat(cy, "AI-Generated", fake_p, (244, 94, 94))

    # ----- Footer -----
    foot = "Forensic analysis — Swin-V2 + FFT + Noise/Edge branches"
    fw, fh = text_size(draw, foot, f_small)
    draw.text((cx, canvas_h - fh - pad // 2), foot,
              font=f_small, fill=text_mut)

    canvas.save(save_path, format="PNG")
    return save_path


# =============================================================================
# INFERENCE
# =============================================================================
@torch.no_grad()
def predict(pil_img, model, device):
    img_rgb = val_tf(pil_img)
    aux     = build_aux(img_rgb)
    logits  = model(img_rgb.unsqueeze(0).to(device), aux.unsqueeze(0).to(device))
    probs   = F.softmax(logits, dim=1).cpu().numpy()[0]
    return int(probs.argmax()), probs


# =============================================================================
# MAIN CALLABLE FUNCTION
# =============================================================================
def run_inference(input_path, ckpt_path=CKPT_PATH, out_dir=OUT_DIR, use_ema=USE_EMA):
    """
    Run forensic inference on one or more images.

    Args:
        input_path: Can be any of:
            - A single image path (str), e.g. "x2.jpeg"
            - A list/tuple of image paths, e.g. ["a.jpg", "b.png"]
            - A folder path, e.g. "images/"
            - A glob pattern, e.g. "imgs/*.png"
        ckpt_path: Path to model checkpoint (default: CKPT_PATH)
        out_dir:   Output directory for reports (default: OUT_DIR)
        use_ema:   Whether to load EMA weights if available (default: USE_EMA)

    Returns:
        List of dicts with keys: path, pred, label, probs, out_path
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SwinForensicModel().to(device)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    use_ema_flag = use_ema and "ema" in ckpt
    state = ckpt["ema"] if use_ema_flag else ckpt["model"]
    print(f"Loaded {'EMA' if use_ema_flag else 'RAW'} weights")
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    # Accept a single path, a list/tuple of paths, a folder, or a glob
    if isinstance(input_path, (list, tuple)):
        files = []
        for p in input_path:
            files.extend(gather_inputs(p))
    else:
        files = gather_inputs(input_path)

    if not files:
        print(f"No images found in: {input_path}")
        return []

    os.makedirs(out_dir, exist_ok=True)
    print(f"Found {len(files)} image(s)\n")

    results = []
    for i, path in enumerate(files, 1):
        try:
            pil = load_image(path)
            pred, probs = predict(pil, model, device)
            stem = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(out_dir, f"{stem}_report.png")
            make_report(pil, probs, pred, out_path)
            print(f"[{i}/{len(files)}] {os.path.basename(path):20s} -> "
                  f"{CLASS_NAMES[pred]:4s}  real={probs[0]*100:6.2f}%  "
                  f"fake={probs[1]*100:6.2f}%  -> {out_path}")
            results.append({
                "path": path,
                "pred": pred,
                "label": CLASS_NAMES[pred],
                "probs": probs,
                "out_path": out_path,
            })
        except Exception as e:
            print(f"[{i}/{len(files)}] {path}  FAILED: {e}")
            results.append({
                "path": path,
                "pred": None,
                "label": None,
                "probs": None,
                "out_path": None,
                "error": str(e),
            })

    return results


if __name__ == "__main__":


    run_inference("x2.jpeg")