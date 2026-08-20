"""Generate the application icon as PNGs, a Windows .ico and a Linux .png.

Drawn at 8x and downsampled, which gives cleaner edges than drawing small.
The motif is deliberately simple -- a selection box with corner handles over a
frame -- because the icon has to stay readable at 16 px in a taskbar.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "icons"
SS = 8  # supersampling factor

BG = (27, 27, 33, 255)
BG_EDGE = (58, 58, 70, 255)
FRAME = (74, 74, 88, 255)
ACCENT = (79, 140, 255, 255)
HANDLE = (255, 255, 255, 255)
GREEN = (67, 192, 122, 255)

ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]
PNG_SIZES = [16, 24, 32, 48, 64, 128, 256, 512]


def _layer(size: int) -> Image.Image:
    return Image.new("RGBA", (size, size), (0, 0, 0, 0))


def draw_icon(size: int) -> Image.Image:
    """Compose the icon at SS x scale, then downsample.

    Translucent fills go on their own layers and are alpha-composited.
    ImageDraw writes RGBA values straight into the buffer instead of blending,
    so drawing a semi-transparent fill directly would punch a hole through the
    background rather than tint it.
    """
    s = size * SS
    base = _layer(s)
    draw = ImageDraw.Draw(base)

    # Rounded background tile.
    radius = int(s * 0.22)
    draw.rounded_rectangle([0, 0, s - 1, s - 1], radius=radius, fill=BG,
                           outline=BG_EDGE, width=max(1, int(s * 0.014)))

    # The "photo" being annotated.
    m = s * 0.155
    draw.rounded_rectangle([m, m, s - m, s - m], radius=int(s * 0.055),
                           outline=FRAME, width=max(1, int(s * 0.020)))

    # Segmentation polygon, low and left so the box does not cover it.
    poly = [(s * 0.235, s * 0.735), (s * 0.375, s * 0.505), (s * 0.515, s * 0.735)]
    fill_layer = _layer(s)
    ImageDraw.Draw(fill_layer).polygon(poly, fill=(GREEN[0], GREEN[1], GREEN[2], 110))
    base = Image.alpha_composite(base, fill_layer)
    ImageDraw.Draw(base).polygon(poly, outline=GREEN, width=max(1, int(s * 0.020)))

    # Detection box, offset up and right of the polygon.
    x1, y1, x2, y2 = s * 0.345, s * 0.245, s * 0.775, s * 0.605
    fill_layer = _layer(s)
    ImageDraw.Draw(fill_layer).rectangle([x1, y1, x2, y2],
                                         fill=(ACCENT[0], ACCENT[1], ACCENT[2], 64))
    base = Image.alpha_composite(base, fill_layer)
    draw = ImageDraw.Draw(base)
    draw.rectangle([x1, y1, x2, y2], outline=ACCENT, width=max(1, int(s * 0.032)))

    # Corner handles: the detail that reads as "annotation tool".
    h = s * 0.042
    for cx, cy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
        draw.rectangle([cx - h, cy - h, cx + h, cy + h], fill=HANDLE,
                       outline=(20, 20, 26, 255), width=max(1, int(s * 0.008)))

    return base.resize((size, size), Image.LANCZOS)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    for size in PNG_SIZES:
        path = OUT / f"yolostudio-{size}.png"
        draw_icon(size).save(path)
    print(f"wrote {len(PNG_SIZES)} PNGs to {OUT}")

    # Windows .ico with every size embedded.
    base = draw_icon(256)
    ico = OUT / "yolostudio.ico"
    base.save(ico, format="ICO",
              sizes=[(n, n) for n in ICO_SIZES])
    print(f"wrote {ico} ({ico.stat().st_size} bytes)")

    # Canonical Linux icon.
    main_png = OUT / "yolostudio.png"
    draw_icon(512).save(main_png)
    print(f"wrote {main_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
