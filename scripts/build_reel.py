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
    import imageio.v2 as iio

    # Lock the reel canvas to the FIRST scenario's frame size; every frame is
    # resized to it so the stream never changes dimensions mid-write.
    W = H = None
    for r in summary:
        paths = sorted((camp / r["scenario_id"] / "frames").glob("frame_*.png"))
        if paths:
            H, W = np.array(Image.open(paths[0])).shape[:2]
            break
    if W is None:
        print("no frames found")
        return 1

    writer = iio.get_writer(out, fps=FPS, codec="libx264", macro_block_size=16,
                            quality=7, ffmpeg_log_level="error")
    n_used = 0
    n_frames = 0
    for r in summary:
        sid = r["scenario_id"]
        paths = sorted((camp / sid / "frames").glob("frame_*.png"))
        if not paths:
            continue
        outcome = "OVERTAKE COMPLETED" if r.get("expected") == "pass" else "HELD / DECLINED (safety)"
        verdict = "PASS" if r.get("success") else "REVIEW"
        card = title_card(W, H, [
            f"{n_used + 1}.  {r.get('town')}  -  {outcome}",
            r.get("narrative", sid),
            f"[{verdict}]  lane_dev<={r.get('max_lane_dev_m')}m  collision={r.get('collision')}",
        ])
        for _ in range(int(TITLE_SECONDS * FPS)):
            writer.append_data(card)
            n_frames += 1
        for p in paths:
            a = np.array(Image.open(p).convert("RGB"))
            if a.shape[:2] != (H, W):
                a = np.array(Image.fromarray(a).resize((W, H)))
            writer.append_data(a)
            n_frames += 1
        n_used += 1
        print(f"  + {sid} ({len(paths)} frames)")
    writer.close()
    print(f"wrote {out}: {n_used} scenarios, {n_frames} frames @ {FPS}fps "
          f"(~{n_frames/FPS:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
