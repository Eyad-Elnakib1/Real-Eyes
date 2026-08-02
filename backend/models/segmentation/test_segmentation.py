"""
AXUNet Inference Script - Styled Heatmap Card Output
Outputs a dark-themed forensic card with the probability heatmap only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import timm
import os
from pathlib import Path

# ============================================================================
# MODEL DEFINITION (Must match training script exactly)
# ============================================================================

class TransformerBottleneck(nn.Module):
    def __init__(self, channels, num_heads=8, num_layers=2):
        super(TransformerBottleneck, self).__init__()
        self.channels = channels
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads,
            dim_feedforward=channels * 4, dropout=0.1,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, channels))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        seq_len = x_flat.size(1)
        x_flat = x_flat + self.pos_embed[:, :seq_len, :]
        x_transformed = self.transformer(x_flat)
        x_out = x_transformed.permute(0, 2, 1).reshape(B, C, H, W)
        return x_out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class NestedDecoderBlock(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(NestedDecoderBlock, self).__init__()
        total_in_channels = sum(in_channels_list)
        self.conv = ConvBlock(total_in_channels, out_channels)

    def forward(self, *inputs):
        target_size = inputs[0].shape[2:]
        resized_inputs = [inputs[0]]
        for inp in inputs[1:]:
            if inp.shape[2:] != target_size:
                inp = F.interpolate(inp, size=target_size, mode='bilinear', align_corners=True)
            resized_inputs.append(inp)
        x = torch.cat(resized_inputs, dim=1)
        return self.conv(x)


class AXUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(AXUNet, self).__init__()
        self.encoder = timm.create_model('xception', pretrained=False, features_only=True)
        encoder_channels = self.encoder.feature_info.channels()  # type: ignore
        self.transformer = TransformerBottleneck(encoder_channels[-1], num_heads=8, num_layers=2)

        dec_channels = [256, 128, 64, 32, 16]

        self.up4 = nn.ConvTranspose2d(encoder_channels[4], dec_channels[0], 2, stride=2)
        self.up3 = nn.ConvTranspose2d(dec_channels[0], dec_channels[1], 2, stride=2)
        self.up2 = nn.ConvTranspose2d(dec_channels[1], dec_channels[2], 2, stride=2)
        self.up1 = nn.ConvTranspose2d(dec_channels[2], dec_channels[3], 2, stride=2)
        self.up0 = nn.ConvTranspose2d(dec_channels[3], dec_channels[4], 2, stride=2)

        self.conv3_1 = NestedDecoderBlock([encoder_channels[3], dec_channels[0]], dec_channels[0])
        self.conv2_1 = NestedDecoderBlock([encoder_channels[2], dec_channels[1]], dec_channels[1])
        self.up3_2 = nn.ConvTranspose2d(dec_channels[0], dec_channels[1], 2, stride=2)
        self.conv2_2 = NestedDecoderBlock([encoder_channels[2], dec_channels[1], dec_channels[1]], dec_channels[1])

        self.conv1_1 = NestedDecoderBlock([encoder_channels[1], dec_channels[2]], dec_channels[2])
        self.up2_2 = nn.ConvTranspose2d(dec_channels[1], dec_channels[2], 2, stride=2)
        self.conv1_2 = NestedDecoderBlock([encoder_channels[1], dec_channels[2], dec_channels[2]], dec_channels[2])
        self.up2_3 = nn.ConvTranspose2d(dec_channels[1], dec_channels[2], 2, stride=2)
        self.conv1_3 = NestedDecoderBlock([encoder_channels[1], dec_channels[2], dec_channels[2], dec_channels[2]], dec_channels[2])

        self.conv0_1 = NestedDecoderBlock([encoder_channels[0], dec_channels[3]], dec_channels[3])
        self.up1_2 = nn.ConvTranspose2d(dec_channels[2], dec_channels[3], 2, stride=2)
        self.conv0_2 = NestedDecoderBlock([encoder_channels[0], dec_channels[3], dec_channels[3]], dec_channels[3])
        self.up1_3 = nn.ConvTranspose2d(dec_channels[2], dec_channels[3], 2, stride=2)
        self.conv0_3 = NestedDecoderBlock([encoder_channels[0], dec_channels[3], dec_channels[3], dec_channels[3]], dec_channels[3])
        self.up1_4 = nn.ConvTranspose2d(dec_channels[2], dec_channels[3], 2, stride=2)
        self.conv0_4 = NestedDecoderBlock([encoder_channels[0], dec_channels[3], dec_channels[3], dec_channels[3], dec_channels[3]], dec_channels[3])

        self.final = nn.Conv2d(dec_channels[3], num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        enc_features = self.encoder(x)
        e0, e1, e2, e3, e4 = enc_features
        e4 = self.transformer(e4)

        d3_1 = self.conv3_1(e3, self.up4(e4))
        d2_1 = self.conv2_1(e2, self.up3(d3_1))
        d1_1 = self.conv1_1(e1, self.up2(d2_1))
        d0_1 = self.conv0_1(e0, self.up1(d1_1))

        d2_2 = self.conv2_2(e2, d2_1, self.up3_2(d3_1))
        d1_2 = self.conv1_2(e1, d1_1, self.up2_2(d2_1))
        d0_2 = self.conv0_2(e0, d0_1, self.up1_2(d1_1))

        d1_3 = self.conv1_3(e1, d1_1, d1_2, self.up2_3(d2_2))
        d0_3 = self.conv0_3(e0, d0_1, d0_2, self.up1_3(d1_2))

        d0_4 = self.conv0_4(e0, d0_1, d0_2, d0_3, self.up1_4(d1_3))

        output = self.final(d0_4)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        output = torch.sigmoid(output)
        return output


# ============================================================================
# INFERENCE UTILITIES
# ============================================================================

def load_model(checkpoint_path, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    model = AXUNet(num_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    print(f"Model loaded (epoch {checkpoint.get('epoch', '?')})")
    return model


def preprocess_image(image, img_size=(512, 768)):
    if isinstance(image, Image.Image):
        image_pil = image.convert('RGB')
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image_pil = Image.fromarray(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.dtype == np.uint8 else image
            ).convert('RGB')
        else:
            image_pil = Image.fromarray(image).convert('RGB')
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Pass a PIL Image or numpy array.")

    original_size = image_pil.size  # (W, H)
    # pyrefly: ignore [missing-attribute]
    image_resized = image_pil.resize((img_size[1], img_size[0]), Image.BILINEAR)
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    return image_tensor.unsqueeze(0), image_pil, original_size


def predict_mask(model, image_tensor, device):
    with torch.no_grad():
        output = model(image_tensor.to(device))
    return output.squeeze().cpu().numpy()


def analyze_regions(prediction, original_size, threshold=0.5, min_area=100):
    pred_resized = cv2.resize(prediction, original_size, interpolation=cv2.INTER_LINEAR)
    binary_mask  = (pred_resized > threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    regions = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        region_mask = (labels == i)
        regions.append({
            'bbox': (int(x), int(y), int(w), int(h)),
            'avg_confidence': float(pred_resized[region_mask].mean()),
        })

    regions.sort(key=lambda r: r['avg_confidence'], reverse=True)
    return regions


# ============================================================================
# STYLED HEATMAP CARD  (dark forensic theme)
# ============================================================================

# ── palette ──────────────────────────────────────────────────────────────────
BG_DARK      = (13,  17,  23)      # #0d1117  outer card
BG_MID       = (17,  26,  24)      # #111a18  header / footer / stat chips
BORDER_COLOR = (30,  45,  42)      # #1e2d2a
TEAL         = (29, 206, 160)      # #1dcea0
TEAL_DIM     = (122, 181, 160)     # #7ab5a0
RED_ACCENT   = (224,  90,  90)     # #e05a5a
WHITE        = (224, 237, 233)     # #e0ede9


def _hex_to_bgr(r, g, b):
    return (b, g, r)


def _load_font(size):
    """Try to load a system monospace font; fall back to Pillow default."""
    candidates = [
        "cour.ttf", "CourierNew.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Courier.ttc",
        "C:/Windows/Fonts/cour.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_rounded_rect(img_cv, x1, y1, x2, y2, radius, color, thickness=-1):
    """Draw a filled or outlined rounded rectangle on a cv2 image."""
    overlay = img_cv.copy()
    # filled rectangles for the body
    if thickness == -1:
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.circle(overlay, (cx, cy), radius, color, -1)
        img_cv[:] = overlay
    else:
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.ellipse(overlay, (cx, cy), (radius, radius), 0, 0, 360, color, thickness)
        img_cv[:] = overlay


def _colorbar(height, width=20):
    """Return a jet-colormap vertical bar as a (height, width, 3) uint8 BGR array."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        t = 1.0 - row / (height - 1)
        color_map = cv2.applyColorMap(
            np.array([[int(t * 255)]], dtype=np.uint8), cv2.COLORMAP_JET
        )
        bar[row, :] = color_map[0, 0]
    return bar


def save_heatmap_card(original_image, prediction, regions, output_path, threshold=0.5):
    """
    Save a single styled PNG card showing the probability heatmap with a
    dark forensic theme matching your app's design.

    Layout (card width = heatmap_w + colorbar + padding):
        ┌─────────────────────────────────┐
        │  HEADER  (label + model badge)  │
        ├─────────────────────────────────┤
        │  heatmap image  │ color bar     │
        ├─────────────────────────────────┤
        │  stat chips: regions / conf /   │
        │  coverage                       │
        ├─────────────────────────────────┤
        │  FOOTER  (model info + verdict) │
        └─────────────────────────────────┘
    """
    img_np = np.array(original_image)
    orig_w, orig_h = img_np.shape[1], img_np.shape[0]

    # ── resize prediction to original size and build jet heatmap ─────────────
    pred_resized  = cv2.resize(prediction, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = (pred_resized * 255).clip(0, 255).astype(np.uint8)
    heatmap_jet   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)   # BGR

    # ── compute stats ─────────────────────────────────────────────────────────
    n_regions  = len(regions)
    max_conf   = regions[0]['avg_confidence'] if regions else 0.0
    coverage   = float((pred_resized > threshold).mean()) * 100
    is_forged  = n_regions > 0 and max_conf > threshold

    # ── layout dimensions ─────────────────────────────────────────────────────
    PADDING      = 20
    CB_WIDTH     = 28     # colorbar strip
    CB_GAP       = 10     # gap between heatmap and colorbar
    HEADER_H     = 52
    FOOTER_H     = 48
    STAT_H       = 64
    INNER_H      = orig_h
    INNER_W      = orig_w

    CARD_W = PADDING + INNER_W + CB_GAP + CB_WIDTH + PADDING
    CARD_H = HEADER_H + PADDING + INNER_H + PADDING + STAT_H + PADDING + FOOTER_H

    card = np.zeros((CARD_H, CARD_W, 3), dtype=np.uint8)
    card[:] = BG_DARK

    # ── helper: draw text with PIL for nicer anti-aliasing ────────────────────
    pil_card = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
    draw     = ImageDraw.Draw(pil_card)

    font_sm  = _load_font(12)
    font_md  = _load_font(15)
    font_lg  = _load_font(22)
    font_xl  = _load_font(28)

    def rgb(r, g, b): return (r, g, b)

    # ── HEADER ────────────────────────────────────────────────────────────────
    # background
    draw.rectangle([0, 0, CARD_W, HEADER_H], fill=rgb(*BG_MID))
    draw.rectangle([0, HEADER_H - 1, CARD_W, HEADER_H], fill=rgb(*BORDER_COLOR))

    # green dot
    draw.ellipse([PADDING, HEADER_H // 2 - 5, PADDING + 10, HEADER_H // 2 + 5],
                 fill=rgb(*TEAL))

    # "DETECTION REPORT" label
    draw.text((PADDING + 18, HEADER_H // 2 - 8), "DETECTION REPORT",
              font=font_md, fill=rgb(*TEAL))

    # model badge (right side)
    badge_text = "AXUNet v1.0"
    bw = font_sm.getlength(badge_text) + 20
    bx = CARD_W - PADDING - int(bw)
    by = HEADER_H // 2 - 11
    draw.rounded_rectangle([bx, by, bx + int(bw), by + 22], radius=4,
                            fill=rgb(15, 42, 34), outline=rgb(*TEAL), width=1)
    draw.text((bx + 10, by + 4), badge_text, font=font_sm, fill=rgb(*TEAL))

    # ── SECTION TITLE ─────────────────────────────────────────────────────────
    title_y = HEADER_H + 10
    draw.text((PADDING, title_y), "Edit Probability Heatmap",
              font=font_sm, fill=rgb(*TEAL_DIM))

    # ── HEATMAP ───────────────────────────────────────────────────────────────
    hm_x = PADDING
    hm_y = HEADER_H + PADDING + 14
    hm_bgr = cv2.cvtColor(np.array(pil_card), cv2.COLOR_RGB2BGR)

    # copy heatmap into card
    hm_bgr[hm_y:hm_y + INNER_H, hm_x:hm_x + INNER_W] = heatmap_jet

    # thin border around heatmap
    cv2.rectangle(hm_bgr,
                  (hm_x - 1, hm_y - 1),
                  (hm_x + INNER_W, hm_y + INNER_H),
                  _hex_to_bgr(*BORDER_COLOR), 1)

    # "PROBABILITY MAP" overlay label
    label_bg_x2 = hm_x + 148
    label_bg_y2 = hm_y + 22
    cv2.rectangle(hm_bgr, (hm_x + 6, hm_y + 6), (label_bg_x2, label_bg_y2),
                  (13, 17, 23), -1)
    cv2.rectangle(hm_bgr, (hm_x + 6, hm_y + 6), (label_bg_x2, label_bg_y2),
                  _hex_to_bgr(*TEAL), 1)

    # ── COLORBAR ──────────────────────────────────────────────────────────────
    cb_x = hm_x + INNER_W + CB_GAP
    cb_y = hm_y
    cb   = _colorbar(INNER_H, CB_WIDTH)
    hm_bgr[cb_y:cb_y + INNER_H, cb_x:cb_x + CB_WIDTH] = cb
    cv2.rectangle(hm_bgr, (cb_x - 1, cb_y - 1),
                  (cb_x + CB_WIDTH, cb_y + INNER_H), _hex_to_bgr(*BORDER_COLOR), 1)

    # ── back to PIL for text ──────────────────────────────────────────────────
    pil_card = Image.fromarray(cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_card)

    # overlay label text
    draw.text((hm_x + 10, hm_y + 8), "PROBABILITY MAP", font=font_sm, fill=rgb(*TEAL))

    # colorbar ticks
    tick_x = cb_x + CB_WIDTH + 4
    draw.text((tick_x, cb_y),                        "1.0", font=font_sm, fill=rgb(*WHITE))
    draw.text((tick_x, cb_y + INNER_H // 2 - 6),    "0.5", font=font_sm, fill=rgb(*WHITE))
    draw.text((tick_x, cb_y + INNER_H - 14),         "0.0", font=font_sm, fill=rgb(*WHITE))

    # ── STAT CHIPS ────────────────────────────────────────────────────────────
    stat_y    = hm_y + INNER_H + PADDING
    chip_w    = (INNER_W - 20) // 3
    chip_data = [
        ("Regions",   str(n_regions),          n_regions > 0),
        ("Max Conf",  f"{max_conf:.2f}",        max_conf > threshold),
        ("Coverage",  f"{coverage:.1f}%",       coverage > 10),
    ]

    for i, (label, value, danger) in enumerate(chip_data):
        cx = hm_x + i * (chip_w + 10)
        cy = stat_y
        draw.rounded_rectangle([cx, cy, cx + chip_w, cy + STAT_H - 8],
                                radius=8, fill=rgb(*BG_MID),
                                outline=rgb(*BORDER_COLOR), width=1)
        draw.text((cx + 10, cy + 8),  label, font=font_sm, fill=rgb(*TEAL_DIM))
        val_color = rgb(*RED_ACCENT) if danger else rgb(*TEAL)
        draw.text((cx + 10, cy + 26), value, font=font_xl, fill=val_color)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    footer_y = CARD_H - FOOTER_H
    draw.rectangle([0, footer_y, CARD_W, CARD_H], fill=rgb(*BG_MID))
    draw.rectangle([0, footer_y, CARD_W, footer_y + 1], fill=rgb(*BORDER_COLOR))

    draw.text((PADDING, footer_y + 16),
              "Forensic analysis — AXUNet + Transformer Bottleneck",
              font=font_sm, fill=rgb(*TEAL_DIM))

    # verdict badge
    verdict_text  = "FORGED" if is_forged else "REAL"
    verdict_color = rgb(*RED_ACCENT) if is_forged else rgb(*TEAL)
    verdict_bg    = rgb(42, 16, 16) if is_forged else rgb(15, 42, 34)
    verdict_brd   = rgb(*RED_ACCENT) if is_forged else rgb(*TEAL)
    vw = int(font_md.getlength(verdict_text)) + 28
    vx = CARD_W - PADDING - vw
    vy = footer_y + 12
    draw.rounded_rectangle([vx, vy, vx + vw, vy + 26], radius=4,
                            fill=verdict_bg, outline=verdict_brd, width=1)
    draw.text((vx + 14, vy + 5), verdict_text, font=font_md, fill=verdict_color)

    # ── outer card rounded border ─────────────────────────────────────────────
    out_cv = cv2.cvtColor(np.array(pil_card), cv2.COLOR_RGB2BGR)
    cv2.rectangle(out_cv, (0, 0), (CARD_W - 1, CARD_H - 1),
                  _hex_to_bgr(*BORDER_COLOR), 2)

    cv2.imwrite(output_path, out_cv)
    print(f"Saved styled heatmap card: {output_path}")


# ============================================================================
# MAIN INFERENCE FUNCTION
# ============================================================================

def detect_forgery(
    image,
    checkpoint_path=str(Path(__file__).resolve().parent / "checkpoint_epoch_21.pth"),
    output_dir="results",
    output_name="result",
    img_size=(512, 768),
    threshold=0.5,
    min_area=100,
):
    """
    Run forgery detection on an image and save a styled heatmap card.

    Args:
        image: PIL Image or numpy array (RGB or BGR uint8)

    Returns:
        dict with keys:
            'regions'     — list of detected regions (bbox, avg_confidence)
            'output_path' — path to the saved styled heatmap card PNG
            'prediction'  — raw numpy prediction mask
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(checkpoint_path, device)

    image_tensor, image_pil, original_size = preprocess_image(image, img_size=img_size)
    prediction = predict_mask(model, image_tensor, device)

    regions = analyze_regions(prediction, original_size,
                               threshold=threshold, min_area=min_area)

    output_path = os.path.join(output_dir, f"{output_name}_heatmap_card.png")
    save_heatmap_card(image_pil, prediction, regions, output_path, threshold=threshold)

    print(f"\n→ {len(regions)} region(s) detected")

    return {
        'regions':      regions,
        'output_path':  output_path,
        'prediction':   prediction,
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    img    = Image.open(r"D:\grad_fainal_work\segmintation_model\s1.png")
    result = detect_forgery(img)
    print(result['regions'])