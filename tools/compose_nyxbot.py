#!/usr/bin/env python3
"""Compose NyxBot's default UI artwork from existing NyxID brand assets.

This is deterministic branding/layout, not AI image generation.
Requires Pillow. Supply a locally installed bold sans-serif TTF with --font.
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--font', type=Path, required=True)
args = parser.parse_args()
assets = Path(__file__).resolve().parents[1] / 'components/NyxPanel/assets'
canvas = Image.open(assets / 'nyxid-default.jpg').convert('RGB')
mark = Image.open(assets / 'nyxid-mark.png').convert('RGBA')
mark.thumbnail((132, 132), Image.Resampling.LANCZOS)
canvas.paste(mark, ((800 - mark.width) // 2, 290), mark)
font = ImageFont.truetype(str(args.font), 88)
draw = ImageDraw.Draw(canvas)
text = 'NyxBot'
bounds = draw.textbbox((0, 0), text, font=font)
draw.text(((800 - bounds[2] + bounds[0]) // 2 - bounds[0], 445), text, font=font, fill='#FFFFFF')
output = assets / 'nyxbot-default.jpg'
for quality in range(90, 39, -5):
    canvas.save(output, quality=quality, optimize=True, progressive=False)
    if output.stat().st_size <= 65536:
        break
assert output.stat().st_size <= 65536
print(output, output.stat().st_size, 'bytes')
