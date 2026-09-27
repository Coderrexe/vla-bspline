#!/usr/bin/env python3
"""Export circular author portraits for both GitHub and the website.

Originals stay untouched. The PNGs contain no source EXIF metadata. A circular
alpha mask works on GitHub too, where inline border-radius styles are stripped.
Requires Pillow. Run from any directory.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "website/public/media/authors"
# Manual head-and-shoulder crops in the supplied image coordinates.
CROPS = {
    "simba": (80, 35, 620, 575),
    "quinten": (275, 255, 525, 505),
    "xiatao": (0, 0, 256, 256),
}


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, box in CROPS.items():
        with Image.open(ROOT / "photos" / f"{name}.jpeg") as original:
            portrait = ImageOps.exif_transpose(original).convert("RGB")
            portrait = portrait.crop(box).resize((256, 256), Image.Resampling.LANCZOS)
        mask = Image.new("L", (1024, 1024), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, 1023, 1023), fill=255)
        portrait.putalpha(mask.resize((256, 256), Image.Resampling.LANCZOS))
        portrait.info.clear()
        destination = DESTINATION / f"{name}.png"
        portrait.save(destination, optimize=True)
        print(f"{destination.relative_to(ROOT)}: {destination.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
