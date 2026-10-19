#!/usr/bin/env python3
"""
Prepare the photo for clean ASCII conversion:
  1. Downscale large photo to reasonable dimensions (~1200px max)
  2. Remove the background (rembg) so the subject is isolated
  3. Bilateral-smooth away noise while keeping facial edges sharp
  4. Stretch tones so skin lands bright and hair/facial features stay clear
  5. Darken line work (difference-of-gaussians ridges)
  6. Composite onto white and square-crop around the subject

Output: source-prepped.png (grayscale), consumed by make_ascii_svg.py.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image
from rembg import remove

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "images", "image.png")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")

LINE_WEIGHT = 0.6     # how hard drawn lines are pushed toward black

print(f"Reading input photo: {INP}")
orig = Image.open(INP)

# Scale down if large for fast, crisp background removal
max_dim = 1200
if max(orig.size) > max_dim:
    ratio = max_dim / max(orig.size)
    new_w, new_h = int(orig.size[0] * ratio), int(orig.size[1] * ratio)
    print(f"Resizing from {orig.size} to {(new_w, new_h)}")
    orig = orig.resize((new_w, new_h), Image.LANCZOS)

# 1. Cut out the subject
print("Removing background with rembg...")
cut = remove(orig.convert("RGBA"))
rgb = np.array(cut.convert("RGB"))
alpha = np.array(cut.split()[-1])                 # 0 = background
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

# 2. Smooth texture, keep edges
print("Applying bilateral filter for edge-preserving smoothing...")
smooth = gray
for _ in range(3):
    smooth = cv2.bilateralFilter(smooth, 9, 40, 9)

# 3. Tone stretch over the subject only
print("Stretching tones...")
if np.any(alpha > 128):
    lo, hi = np.percentile(smooth[alpha > 128], [2, 92])
    if hi > lo:
        tone = np.clip((smooth.astype(np.float32) - lo) / (hi - lo), 0, 1)
    else:
        tone = smooth.astype(np.float32) / 255.0
else:
    tone = smooth.astype(np.float32) / 255.0

# 4. Dark-on-light ridges -> darken
print("Enhancing edge contours...")
fine = cv2.GaussianBlur(smooth, (0, 0), 1.5).astype(np.float32)
coarse = cv2.GaussianBlur(smooth, (0, 0), 6).astype(np.float32)
lines = np.clip((coarse - fine) / 40.0, 0, 1)
out = np.clip(tone - LINE_WEIGHT * lines, 0, 1) * 255

# 5. Paste onto white, square crop
print("Compositing onto white and cropping...")
mask = cv2.GaussianBlur(alpha.astype(np.float32) / 255.0, (0, 0), 1.0)
out = out * mask + 255.0 * (1.0 - mask)

ys, xs = np.where(alpha > 20)
if len(xs) > 0 and len(ys) > 0:
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) + 60
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    canvas = np.full((side, side), 255, np.uint8)
    x0, y0 = cx - side // 2, cy - side // 2
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, out.shape[1]), min(y0 + side, out.shape[0])
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = out[sy0:sy1, sx0:sx1].astype(np.uint8)
else:
    canvas = out.astype(np.uint8)

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
Image.fromarray(canvas, mode="L").save(OUT)
print(f"Successfully wrote {OUT} (dimensions: {canvas.shape})")
