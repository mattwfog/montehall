"""Synthetic jersey-digit crop generator (license-clean training data).

Renders 0-99 on jersey-like backgrounds with the degradations that make
real torso crops hard: perspective tilt, motion blur, low resolution,
fabric-fold shading, partial occlusion. Output feeds the PARSeq jersey
fine-tune and the legibility gate (a crop rendered illegible on purpose
is a negative sample with a known label).

Pure numpy + PIL. Deterministic per seed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

JERSEY_PALETTES = [
    ((250, 250, 250), (20, 20, 90)),    # white / navy digits
    ((30, 30, 30), (240, 240, 240)),    # black / white
    ((180, 30, 40), (255, 255, 255)),   # red / white
    ((30, 60, 160), (250, 210, 60)),    # blue / gold
    ((20, 110, 60), (255, 255, 255)),   # green / white
    ((250, 200, 40), (40, 40, 120)),    # gold / navy
    ((110, 40, 130), (250, 250, 250)),  # purple / white
    ((230, 230, 235), (180, 30, 40)),   # grey / red
]


def generate(out_dir: Path, n: int, seed: int = 0, illegible_frac: float = 0.25,
             hard: bool = False) -> dict:
    """hard=True widens ranges toward real far-court torso crops: smaller
    bases, stronger perspective, heavier still-legible degradation, mild
    edge cuts. Illegible generation is unchanged (already the hard bucket)."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels: list[dict] = []
    fonts = _fonts()

    for i in range(n):
        number = str(rng.randint(0, 99))
        jersey_rgb, digit_rgb = rng.choice(JERSEY_PALETTES)
        make_illegible = rng.random() < illegible_frac

        if hard:
            w, h = rng.randint(32, 140), rng.randint(28, 120)
        else:
            w, h = rng.randint(56, 140), rng.randint(48, 120)
        img = Image.new("RGB", (w * 3, h * 3), jersey_rgb)
        draw = ImageDraw.Draw(img)
        _fabric_shading(img, np_rng)

        font_path = rng.choice(fonts)
        font_size = int(h * 3 * rng.uniform(0.45, 0.72))
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            font = ImageFont.load_default(size=font_size)
        bbox = draw.textbbox((0, 0), number, font=font)
        tx = (img.width - (bbox[2] - bbox[0])) // 2 - bbox[0] + rng.randint(-w // 3, w // 3)
        ty = (img.height - (bbox[3] - bbox[1])) // 2 - bbox[1] + rng.randint(-h // 3, h // 3)
        outline_rgb = tuple(255 - c for c in digit_rgb)
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            draw.text((tx + dx, ty + dy), number, font=font, fill=outline_rgb)
        draw.text((tx, ty), number, font=font, fill=digit_rgb)

        img = _perspective(img, rng, jitter_max=0.22 if hard else 0.14)
        if not make_illegible:
            img = _degrade_light(img, rng, hard=hard)
        img = img.resize((w, h), Image.BILINEAR)
        if make_illegible:
            # degrade AFTER downsizing — blur at final scale destroys strokes;
            # pre-resize blur survives the resample and stays readable
            img = _degrade_hard(img, rng)
        elif hard:
            img = _edge_cut(img, rng)

        name = f"{i:06d}.jpg"
        img.save(out_dir / name, quality=rng.randint(40 if hard else 55, 92))
        labels.append(
            {"file": name, "label": number if not make_illegible else None,
             "legible": not make_illegible}
        )

    with open(out_dir / "labels.jsonl", "w") as f:
        for row in labels:
            f.write(json.dumps(row) + "\n")
    n_legible = sum(1 for r in labels if r["legible"])
    return {"generated": n, "legible": n_legible, "illegible": n - n_legible}


def _fonts() -> list[str]:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    found = [p for p in candidates if Path(p).exists()]
    return found or [""]


def _fabric_shading(img, np_rng) -> None:
    arr = np.asarray(img, dtype=np.float32)
    yy = np.linspace(0, 4 * np.pi, arr.shape[0])[:, None]
    folds = 12 * np.sin(yy + np_rng.uniform(0, np.pi)) * np_rng.uniform(0.3, 1.0)
    arr += folds[..., None]
    np.copyto(arr, arr.clip(0, 255))
    img.paste(_from_array(arr))


def _from_array(arr: np.ndarray):
    from PIL import Image

    return Image.fromarray(arr.astype(np.uint8))


def _perspective(img, rng, jitter_max: float = 0.14):
    from PIL import Image

    w, h = img.size
    jitter = int(w * rng.uniform(0.02, jitter_max))
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (rng.randint(0, jitter), rng.randint(0, jitter)),
        (w - rng.randint(0, jitter), rng.randint(0, jitter)),
        (w - rng.randint(0, jitter), h - rng.randint(0, jitter)),
        (rng.randint(0, jitter), h - rng.randint(0, jitter)),
    ]
    coeffs = _perspective_coeffs(src, dst)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)


def _perspective_coeffs(src, dst):
    matrix = []
    for (sx, sy), (dx, dy) in zip(src, dst, strict=True):
        matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray([c for pair in src for c in pair], dtype=np.float64)
    return np.linalg.lstsq(a, b, rcond=None)[0].tolist()


def _degrade_light(img, rng, hard: bool = False):
    from PIL import ImageFilter

    if rng.random() < (0.75 if hard else 0.6):
        sigma_hi = 2.1 if hard else 1.4  # stays under _degrade_hard's 2.2 floor
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, sigma_hi)))
    if hard and rng.random() < 0.5:  # sensor noise on dim gym footage
        arr = np.asarray(img, dtype=np.float32)
        arr += np.random.default_rng(rng.randrange(2**32)).normal(
            0, rng.uniform(3, 9), arr.shape)
        img = _from_array(arr.clip(0, 255))
    return img


def _edge_cut(img, rng):
    """Mild off-center reframe (<=15% per side) — digits stay fully readable;
    the '52'-cropped-to-'5' ambiguity class stays in the illegible bucket."""
    from PIL import Image

    w, h = img.size
    dx = int(w * rng.uniform(0, 0.15)) * rng.choice([-1, 1])
    canvas = Image.new("RGB", (w, h), img.getpixel((2, 2)))
    canvas.paste(img, (dx, 0))
    return canvas


def _degrade_hard(img, rng):
    from PIL import Image, ImageDraw, ImageFilter

    w, h = img.size
    mode = rng.choice(["blur", "occlude", "tiny", "crop"])
    if mode == "blur":
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(2.2, 4.5)))
    elif mode == "occlude":  # limb/arm across the number
        draw = ImageDraw.Draw(img)
        x = rng.randint(w // 6, w // 2)
        draw.rectangle(
            [x, 0, x + rng.randint(w // 3, (2 * w) // 3), h], fill=(120, 90, 70)
        )
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 1.8)))
    elif mode == "tiny":  # far-court crop: downscale through 10-16px and back
        tiny_h = rng.randint(10, 16)
        tiny_w = max(int(w * tiny_h / h), 8)
        img = img.resize((tiny_w, tiny_h), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    else:  # number half out of frame
        shift = rng.randint(w // 3, (2 * w) // 3) * rng.choice([-1, 1])
        canvas = Image.new("RGB", (w, h), img.getpixel((2, 2)))
        canvas.paste(img, (shift, 0))
        img = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 2.0)))
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hard", action="store_true",
                        help="ranges tuned toward real far-court torso crops")
    args = parser.parse_args()
    print(json.dumps(generate(args.out, args.n, args.seed, hard=args.hard), indent=2))


if __name__ == "__main__":
    main()
