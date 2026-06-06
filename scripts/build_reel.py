#!/usr/bin/env python3
"""Stitch the per-scenario clips into one presentation reel with title cards.

Usage: python scripts/build_reel.py <campaign_dir> [out.mp4]
Reads <campaign_dir>/summary.json and each scenario's frames/, prepends a title card
with the scenario narrative + result, and concatenates into one MP4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 18
TITLE_SECONDS = 2.2


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def title_card(w: int, h: int, lines, color=(255, 255, 255)) -> np.ndarray:
    img = Image.new("RGB", (w, h), (12, 14, 20))
    d = ImageDraw.Draw(img)
    fonts = [_font(38), _font(26), _font(22)]
    y = int(h * 0.30)
    for i, line in enumerate(lines):
        f = fonts[min(i, len(fonts) - 1)]
        col = color if i == 0 else (200, 210, 225) if i == 1 else (150, 200, 150)
        try:
            tw = d.textlength(line, font=f)
        except Exception:
            tw = len(line) * 12
        d.text(((w - tw) / 2, y), line, fill=col, font=f)
        y += int(f.size * 1.6)
    return np.array(img)


def main() -> int:
    camp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else camp / "presentation_reel.mp4"
    summary = json.loads((camp / "summary.json").read_text(encoding="utf-8"))
    import imageio.v3 as iio

    frames: list[np.ndarray] = []
    W = H = None
    n_used = 0
    for r in summary:
        sid = r["scenario_id"]
        fdir = camp / sid / "frames"
        paths = sorted(fdir.glob("frame_*.png"))
        if not paths:
            continue
        sample = np.array(Image.open(paths[0]))
        H, W = sample.shape[:2]
        outcome = "OVERTAKE COMPLETED" if r.get("expected") == "pass" else "HELD / DECLINED (safety)"
        verdict = "PASS" if r.get("success") else "REVIEW"
        card = title_card(W, H, [
            f"{n_used + 1}.  {r.get('town')}  -  {outcome}",
            r.get("narrative", sid),
            f"[{verdict}]  lane_dev<={r.get('max_lane_dev_m')}m  collision={r.get('collision')}",
        ])
        for _ in range(int(TITLE_SECONDS * FPS)):
            frames.append(card)
        for p in paths:
            a = np.array(Image.open(p))
            if a.shape[:2] != (H, W):
                a = np.array(Image.fromarray(a).resize((W, H)))
            frames.append(a)
        n_used += 1

    if not frames:
        print("no frames found")
        return 1
    # macro-block safe sizing
    iio.imwrite(out, frames, fps=FPS, codec="libx264", macro_block_size=16)
    print(f"wrote {out}: {n_used} scenarios, {len(frames)} frames @ {FPS}fps "
          f"(~{len(frames)/FPS:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
