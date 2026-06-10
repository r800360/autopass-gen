"""Build the three presentation figures with robust fonts, no em dashes, large text.
Outputs: presentation_assets/fig_architecture.png, fig_problem.png, fig_results.png
"""
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image, ImageDraw, ImageFont

NAVY = "#1F3A5F"; TEAL = "#1390A6"; AMBER = "#E08A2B"; GREEN = "#2BA84A"; RED = "#D2433A"
GRAY = "#5B6B7B"; INK = "#16222E"; LNAVY = "#EAF1F8"; ROAD = "#6E7780"; BLUE = "#2477C8"

_FT = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")


def pil_font(sz, bold=True):
    p = os.path.join(_FT, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(p, sz)
    except Exception:
        try:
            return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", sz)
        except Exception:
            return ImageFont.load_default()


# ===================== ARCHITECTURE =====================
def build_architecture():
    fig, ax = plt.subplots(figsize=(10.8, 4.2), dpi=210)
    ax.set_xlim(0, 112); ax.set_ylim(0, 42); ax.axis("off")

    def box(x, y, w, h, title, sub, fc, tc="white", fs=11, subfs=8.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2.2", fc=fc, ec="none", zorder=3))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold", zorder=4)
        if sub:
            ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", color=tc, fontsize=subfs, zorder=4)

    def arrow(p1, p2, color=INK, lw=2.2, rad=0.0):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=2))

    y, h = 23, 9.5
    box(2, y, 18, h, "PERCEPTION", "front+rear\nseg + depth  ->  gaps", TEAL)
    box(24, y, 18, h, "PLANNER  (LLM)", "chooses next tool\neach cycle", NAVY)
    box(46, y, 18, h, "CRITIC", "deterministic\nsafety verify", AMBER)
    box(72, y, 18, h, "CONTROLLER", "waypoint lane-keep\n+ lane change", GREEN)
    box(94, y, 16, h, "CARLA", "execute +\nobserve", GRAY)

    arrow((20, y + h / 2), (24, y + h / 2))
    arrow((42, y + h / 2), (46, y + h / 2))
    arrow((64, y + h / 2), (72, y + h / 2), color=GREEN)
    ax.text(68, y + h / 2 + 2.4, "safe", ha="center", color=GREEN, fontsize=9, fontweight="bold")
    arrow((90, y + h / 2), (94, y + h / 2), color=GREEN)

    # reject arc: critic -> planner
    arrow((50, y), (34, y), color=RED, rad=-0.45, lw=2.2)
    ax.text(44.07, y - 5.0, "unsafe  ->  re-sense / replan", ha="center", color=RED, fontsize=8.8, fontweight="bold")

    # clean rectangular feedback bus over the top (no catenary sag)
    ytop = y + h + 4.5
    ccx, pcx = 102, 11
    # ax.plot([ccx, ccx, pcx, pcx], [y + h, ytop, ytop, y + h], color=GRAY, lw=2.0, ls=(0, (6, 3)), zorder=1)
    ax.plot([ccx, ccx, pcx, pcx], [y + h, ytop, ytop, y + h], color=GRAY, lw=2.0, zorder=1)
    ax.add_patch(FancyArrowPatch((pcx, ytop), (pcx, y + h), arrowstyle="-|>", mutation_scale=15, lw=2.0, color=GRAY, zorder=2))
    ax.text((ccx + pcx) / 2, ytop + 1.4, "closed loop:   observe result   ->   update belief", ha="center", color=GRAY, fontsize=9.2, fontweight="bold")

    # shared DSL bar
    dx, dw, dy, dh = 24, 66, 8, 7.5
    ax.add_patch(FancyBboxPatch((dx, dy), dw, dh, boxstyle="round,pad=0.6,rounding_size=2.2", fc=LNAVY, ec=NAVY, lw=1.6, zorder=3))
    ax.text(dx + dw / 2, dy + dh * 0.64, "SHARED  DSL  (mutable)", ha="center", va="center", color=NAVY, fontsize=10.5, fontweight="bold", zorder=4)
    ax.text(dx + dw / 2, dy + dh * 0.26, "belief  +  memory (denials, tool history)  +  plan", ha="center", va="center", color=INK, fontsize=8.4, zorder=4)
    for cx in (33, 55, 81):
        ax.add_patch(FancyArrowPatch((cx, dy + dh), (cx, y), arrowstyle="<|-|>", mutation_scale=11, lw=1.5, color=NAVY, zorder=2))

    tools = ["sense_front", "sense_rear", "sense_passing_lane", "check_corridor", "propose_pass", "hold"]
    ax.text(2, 3.3, "Tool palette:", color=INK, fontsize=8.8, fontweight="bold", va="center")
    tx = 16.5
    for t in tools:
        w = len(t) * 0.92 + 3.2
        ax.add_patch(FancyBboxPatch((tx, 1.6), w, 3.4, boxstyle="round,pad=0.2,rounding_size=1.4", fc="white", ec=NAVY, lw=1.2, zorder=3))
        ax.text(tx + w / 2, 3.3, t, ha="center", va="center", color=NAVY, fontsize=7.6, zorder=4)
        tx += w + 1.6

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig("presentation_assets/fig_architecture.png", dpi=210, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close()
    print("fig_architecture.png")


# ===================== PROBLEM =====================
def build_problem():
    fig, ax = plt.subplots(figsize=(10.6, 3.95), dpi=210)
    ax.set_xlim(0, 112); ax.set_ylim(0, 42); ax.axis("off")
    ax.add_patch(plt.Rectangle((4, 9), 104, 22, fc=ROAD, ec="none", zorder=1))
    for yy in (9, 31):
        ax.plot([4, 108], [yy, yy], color="white", lw=2.2, zorder=2)
    for x0 in range(6, 108, 7):
        ax.plot([x0, x0 + 4], [20, 20], color="white", lw=2.0, zorder=2)

    def car(x, y, w, hh, fc, label):
        ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.2,rounding_size=1.6", fc=fc, ec="white", lw=1.4, zorder=4))
        ax.text(x + w / 2, y + hh / 2, label, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold", zorder=5)

    car(70, 11.5, 11, 6, RED, "LEAD")
    car(34, 11.5, 11, 6, BLUE, "EGO")
    ax.text(75.5, 8.0, "slow", ha="center", color=RED, fontsize=10, fontweight="bold")
    car(20, 22.5, 11, 6, "#33414E", "FAST")
    ax.add_patch(FancyArrowPatch((46, 14.5), (58, 14.5), arrowstyle="-|>", mutation_scale=14, lw=2, color="white", zorder=5))
    ax.add_patch(FancyArrowPatch((32, 25.5), (52, 25.5), arrowstyle="-|>", mutation_scale=16, lw=2.6, color="#FFD24A", zorder=5))
    ax.text(60, 25.8, "closing fast", ha="left", va="center", color="#FFD24A", fontsize=9.5, fontweight="bold", zorder=5)
    ax.add_patch(Circle((100, 37), 3.4, fc=AMBER, ec="none", zorder=4))
    ax.text(100, 37, "!", ha="center", va="center", color="white", fontsize=15, fontweight="bold", zorder=5)
    ax.text(95.5, 37, "trip deadline", ha="right", va="center", color=INK, fontsize=10.5, fontweight="bold", zorder=5)
    ax.text(56, 38.6, "Overtake now, or wait for the lane to clear?", ha="center", va="center", color=NAVY, fontsize=14, fontweight="bold")
    ax.text(56, 3.4, "Safe only if the ego reads the REAR and ONCOMING lanes first, not from a pre-planned trajectory.",
            ha="center", va="center", color=INK, fontsize=10.2)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig("presentation_assets/fig_problem.png", dpi=210, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close()
    print("fig_problem.png")


# ===================== RESULTS =====================
def bev_crop(sid, frac):
    paths = sorted(glob.glob(f"runs/campaign_agentic_v2/{sid}/frames/frame_*.png"))
    im = Image.open(paths[int(len(paths) * frac)]).convert("RGB")
    w, h = im.size
    return im.crop((w // 2 + 2, 0, w, h))


def build_results():
    NAVYc = (31, 58, 95); INKc = (22, 34, 46); GRAYc = (91, 107, 123); GREENc = (31, 130, 60)
    W, H = 2200, 840
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    # header
    d.text((W // 2, 60), "Generalization across 4 CARLA towns and many conditions",
           font=pil_font(54), fill=NAVYc, anchor="mm")
    # montage strip
    thumbs = [("s01_t04_highway_safe_pass", 0.60, "Highway"),
              ("s13_t01_rural_two_lane_pass", 0.62, "Rural two-lane"),
              ("s09_t03_urban_pass", 0.62, "Urban"),
            #   ("s22_t04_highway_heavy_traffic_pass", 0.60, "Heavy traffic"),
              ("s23_t04_highway_truck_pass", 0.60, "Truck lead")]
    m = len(thumbs); side = 55; gap = 26
    tw = (W - 2 * side - (m - 1) * gap) // m
    th = int(tw * 3 / 4)
    y0 = 150
    capf = pil_font(40)
    for i, (sid, frac, cap) in enumerate(thumbs):
        x = side + i * (tw + gap)
        try:
            im = bev_crop(sid, frac).resize((tw, th))
        except Exception:
            im = Image.new("RGB", (tw, th), (40, 40, 40))
        canvas.paste(im, (x, y0))
        d.rectangle([x, y0, x + tw, y0 + th], outline=NAVYc, width=4)
        d.text((x + tw // 2, y0 + th + 34), cap, font=capf, fill=INKc, anchor="mm")
    cy = y0 + th + 110
    d.text((W // 2, cy), "clear / wet / hard-rain / dusk / overcast      left and right passes      live ambient traffic",
           font=pil_font(40, bold=False), fill=GRAYc, anchor="mm")
    d.text((W // 2, cy + 64), "Every pass, wait, and yield-then-pass decision came from the live agentic loop.",
           font=pil_font(42), fill=GREENc, anchor="mm")
    canvas.save("presentation_assets/fig_results.png")
    print("fig_results.png", canvas.size)


if __name__ == "__main__":
    build_architecture()
    build_problem()
    build_results()
