"""Rebuild the slide-5 demo clip with a LARGE, readable caption banner (from the trace)
plus large title cards. No CARLA needed - post-processes existing v2 frames."""
import json
import glob
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as iio

CAMP = "runs/campaign_agentic_v2"
FPS = 9
BH = 304            # banner height (720 + 304 = 1024, both divisible by 16)
CARD_S = 1.5        # seconds per title card
OUT = "presentation_assets/demo_hero.mp4"

AMBER = (236, 152, 42)
GREEN = (46, 184, 84)
BLUE = (78, 178, 214)
WHITE = (238, 242, 247)
TEAL = (70, 200, 200)
BG = (13, 19, 27)


import matplotlib
_FT = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")


def font(sz, bold=True):
    p = os.path.join(_FT, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(p, sz)
    except Exception:
        try:
            return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", sz)
        except Exception:
            return ImageFont.load_default()


def friendly(blk):
    b = " ".join(blk).lower()
    if "rear" in b:
        return "fast vehicle in the passing lane"
    if "oncoming" in b:
        return "oncoming traffic in the passing lane"
    if "front" in b:
        return "gap ahead not safe yet"
    return "reading the passing lane"


def add_banner(im, label, reason, color):
    W, H = im.size
    out = Image.new("RGB", (W, H + BH), BG)
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    d.line([(0, H + 3), (W, H + 3)], fill=color, width=6)
    d.text((52, H + int(BH * 0.34)), label, font=font(146), fill=color, anchor="lm")
    d.text((54, H + int(BH * 0.76)), reason, font=font(76, bold=False), fill=WHITE, anchor="lm")
    return out


def card(title, sub, tag, W, H):
    im = Image.new("RGB", (W, H), (15, 27, 44))
    d = ImageDraw.Draw(im)
    d.text((W / 2, H * 0.30), tag, font=font(86), fill=TEAL, anchor="mm")
    d.text((W / 2, H * 0.50), title, font=font(100), fill=WHITE, anchor="mm")
    d.text((W / 2, H * 0.70), sub, font=font(74, bold=False), fill=(176, 190, 206), anchor="mm")
    return im


def frame_state(tr, t_commit, t):
    if t >= t_commit - 1e-6:
        return "OVERTAKING", "passing lane clear", GREEN
    cur = tr[0]
    for e in tr:
        if e["t_s"] <= t + 1e-6:
            cur = e
        else:
            break
    blk = cur.get("gates", {}).get("blockers", [])
    if blk:
        return "WAITING", friendly(blk), AMBER
    return "ASSESSING", "reading the passing lane", BLUE


# determine canvas size from first frame
sample = Image.open(sorted(glob.glob(f"{CAMP}/s17_t04_reject_rear_traffic/frames/frame_*.png"))[0])
W = sample.width
H = sample.height + BH

writer = iio.get_writer(OUT, fps=FPS, codec="libx264", macro_block_size=16,
                        quality=8, ffmpeg_log_level="error")


def write_card(title, sub, tag):
    img = np.array(card(title, sub, tag, W, H))
    for _ in range(int(CARD_S * FPS)):
        writer.append_data(img)


def write_scenario(sid):
    tr = [e for e in json.load(open(f"{CAMP}/{sid}/trace.json")) if "t_s" in e]
    t_commit = tr[-1]["t_s"]
    frames = sorted(glob.glob(f"{CAMP}/{sid}/frames/frame_*.png"))
    poster_path = None
    for i, fp in enumerate(frames):
        lab, rea, col = frame_state(tr, t_commit, i * 0.1)
        im = add_banner(Image.open(fp).convert("RGB"), lab, rea, col)
        writer.append_data(np.array(im))
        if poster_path is None and lab == "WAITING":
            im.save("presentation_assets/demo_poster.png")
            poster_path = True


write_card("Yield, then overtake", "Highway: a fast car is closing in the passing lane", "Scenario 1")
write_scenario("s17_t04_reject_rear_traffic")
write_card("Wait for oncoming, then overtake", "Rural two-lane: oncoming traffic in the only passing lane", "Scenario 2")
write_scenario("s19_t01_rural_oncoming_reject")
writer.close()
print("wrote", OUT, f"({W}x{H})")
