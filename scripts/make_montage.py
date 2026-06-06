#!/usr/bin/env python3
"""Build a vertical montage of evenly-spaced frames from a run, for quick visual review.

Usage: python scripts/make_montage.py <frames_dir> <out.png> [n]
Stacks n rows (default 6), each row = one sampled composite frame, downscaled.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    frames_dir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    paths = sorted(frames_dir.glob("frame_*.png"))
    if not paths:
        print(f"no frames in {frames_dir}")
        return 1
    idxs = [round(i * (len(paths) - 1) / (n - 1)) for i in range(n)] if len(paths) > 1 else [0]
    rows = []
    scale_w = 900
    for k in idxs:
        im = Image.open(paths[k])
        w, h = im.size
        nh = max(1, round(h * scale_w / w))
        im = im.resize((scale_w, nh))
        a = np.array(im)
        # label the frame index
        rows.append(a)
    montage = np.concatenate(rows, axis=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(montage).save(out)
    print(f"wrote {out} ({len(idxs)} rows from {len(paths)} frames: {idxs})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
