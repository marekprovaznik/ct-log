import argparse
import base64
import html
import io
import json
import pathlib
from typing import List, Optional, Tuple
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FRAMES: List[Tuple[str, str, str]] = [
    ("3", "page_036", ""),
    ("3", "page_057", ""),
    ("3", "page_059", ""),
    ("3", "page_060", ""),
    ("3", "page_069", ""),
    ("3", "page_077", ""),
    ("3", "page_085", "85+ = pěkné (from here on it looks good)"),
    ("3", "page_102", "kůra (bark)"),
    ("3", "page_121", "121–125"),
    ("3", "page_122", "121–125"),
    ("3", "page_123", "121–125"),
    ("3", "page_124", "121–125"),
    ("3", "page_125", "121–125"),
    ("3", "page_169", ""),
    ("3", "page_192", "dvojitá dřeň? (double pith?)"),
    ("3", "page_193", "dvojitá dřeň? (double pith?)"),
    ("3", "page_256", ""),
    ("09", "page_199", "kůra (bark)"),
    ("50", "page_207", "207+ zárost? (overgrowth?)"),
    ("50", "page_208", "207+"),
    ("50", "page_209", "207+"),
    ("50", "page_210", "207+"),
    ("50", "page_211", "207+"),
    ("50", "page_212", "207+"),
]
M4_COLOR = (0, 220, 255)
M5_COLOR = (255, 0, 255)
BOTH_COLOR = (255, 215, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side M4 vs M5 pre-label gallery for selected frames.")
    parser.add_argument("--m4_roots", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--m5_roots", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--crop", type=int, default=0, help="Crop to the wood bounding box plus this margin (0 = full slice)."
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for log, page, note in FRAMES:
        m4 = find(args.m4_roots, log, page)
        m5 = find(args.m5_roots, log, page)
        if m4 is None or m5 is None:
            rows.append((log, page, note, None, "missing: %s" % ("M4" if m4 is None else "M5")))
            continue
        tile, stats = render(m4, m5, args.crop)
        name = "log%s_%s.png" % (log, page)
        tile.save(args.out / name)
        rows.append((log, page, note, name, stats))
        print(log, page, stats)
    (args.out / "index.html").write_text(index_html(rows))


def find(roots: List[pathlib.Path], log: str, page: str) -> Optional[Tuple[pathlib.Path, pathlib.Path]]:
    for root in roots:
        img = root / log / "img" / ("%s.tiff" % page)
        ann = root / log / "ann" / ("%s.tiff.json" % page)
        if img.exists() and ann.exists():
            return img, ann
    return None


def render(
    m4: Tuple[pathlib.Path, pathlib.Path], m5: Tuple[pathlib.Path, pathlib.Path], margin: int
) -> Tuple[Image.Image, str]:
    gray = np.array(Image.open(m4[0]))
    gray = gray[..., 0] if gray.ndim == 3 else gray
    k4, w4, p4 = read_ann(m4[1], gray.shape)
    k5, w5, p5 = read_ann(m5[1], gray.shape)
    box = (0, 0, gray.shape[1], gray.shape[0])
    if margin:
        ys, xs = np.nonzero(w4 | w5)
        box = (
            max(xs.min() - margin, 0),
            max(ys.min() - margin, 0),
            min(xs.max() + margin, gray.shape[1]),
            min(ys.max() + margin, gray.shape[0]),
        )
    base = np.stack([gray] * 3, -1).astype(np.uint8)
    panels = [
        label(Image.fromarray(base), "raw"),
        label(overlay(base, [(k4, M4_COLOR)], [(p4, M4_COLOR)]), "M4 (current)"),
        label(overlay(base, [(k5, M5_COLOR)], [(p5, M5_COLOR)]), "M5 (new)"),
        label(
            overlay(
                base,
                [(k4 & k5, BOTH_COLOR), (k4 & ~k5, M4_COLOR), (k5 & ~k4, M5_COLOR)],
                [(p4, M4_COLOR), (p5, M5_COLOR)],
            ),
            "both yellow · M4 only cyan · M5 only magenta",
        ),
    ]
    panels = [p.crop(box) for p in panels]
    w, h = panels[0].size
    tile = Image.new("RGB", (4 * w + 30, h), "white")
    for i, p in enumerate(panels):
        tile.paste(p, (i * (w + 10), 0))
    stats = "knot px M4 %d · M5 %d · M4 only %d · M5 only %d · pith shift %s" % (
        k4.sum(),
        k5.sum(),
        (k4 & ~k5).sum(),
        (k5 & ~k4).sum(),
        pith_shift(p4, p5),
    )
    return tile, stats


def read_ann(path: pathlib.Path, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    ann = json.loads(path.read_text())
    knots = np.zeros(shape, bool)
    wood = np.zeros(shape, bool)
    pith = None
    for obj in ann["objects"]:
        if obj["geometryType"] == "bitmap":
            raw = zlib.decompress(base64.b64decode(obj["bitmap"]["data"]))
            mask = np.array(Image.open(io.BytesIO(raw)))
            mask = mask[..., -1] if mask.ndim == 3 else mask
            x0, y0 = obj["bitmap"]["origin"]
            h, w = mask.shape
            target = knots if obj["classTitle"] == "Knot" else wood
            target[y0 : y0 + h, x0 : x0 + w] |= mask > 0
        elif obj["classTitle"] == "Pith":
            x, y = obj["points"]["exterior"][0]
            pith = (int(x), int(y))
    return knots, wood, pith


def overlay(
    base: np.ndarray,
    masks: List[Tuple[np.ndarray, Tuple[int, int, int]]],
    piths: List[Tuple[Optional[Tuple[int, int]], Tuple[int, int, int]]],
) -> Image.Image:
    out = base.astype(np.float32).copy()
    for mask, color in masks:
        out[mask] = 0.45 * out[mask] + 0.55 * np.array(color)
    img = Image.fromarray(out.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for pith, color in piths:
        if pith is not None:
            x, y = pith
            draw.line([(x - 6, y - 6), (x + 6, y + 6)], fill=color, width=2)
            draw.line([(x - 6, y + 6), (x + 6, y - 6)], fill=color, width=2)
    return img


def label(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=16)
    draw.rectangle([0, 0, 8 + 9 * len(text), 24], fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255), font=font)
    return img


def pith_shift(p4: Optional[Tuple[int, int]], p5: Optional[Tuple[int, int]]) -> str:
    if p4 is None or p5 is None:
        return "n/a"
    return "%.1f px" % float(np.hypot(p4[0] - p5[0], p4[1] - p5[1]))


def index_html(rows: List[Tuple[str, str, str, Optional[str], str]]) -> str:
    parts = [
        "<!doctype html><meta charset='utf-8'><title>M4 vs M5 on Rosťa's frames</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#f4f4f2;color:#222}figure{margin:0 0 28px}img{max-width:100%;border-radius:4px}figcaption{margin-top:4px;font-size:14px}h2{margin:24px 0 6px;font-size:18px}.n{color:#a05a00}</style>",
        "<h1>M4 vs M5 pre-labels on frames Rosťa flagged</h1>",
        "<p>Panels: raw · M4 (cyan, current preview 384366) · M5 (magenta, ct-log-v5-preview 384679) · difference (yellow both, cyan M4 only, magenta M5 only). × = predicted pith. No human labels exist for these logs.</p>",
    ]
    current = None
    for log, page, note, name, stats in rows:
        if log != current:
            parts.append("<h2>Log %s</h2>" % html.escape(log))
            current = log
        cap = "<b>%s</b> %s<br>%s" % (
            html.escape(page),
            ("<span class='n'>Rosťa: %s</span>" % html.escape(note)) if note else "",
            html.escape(stats),
        )
        parts.append(
            "<figure>%s<figcaption>%s</figcaption></figure>"
            % ("<a href='%s'><img src='%s' loading='lazy'></a>" % (name, name) if name else "", cap)
        )
    return "\n".join(parts)


if __name__ == "__main__":
    main()
