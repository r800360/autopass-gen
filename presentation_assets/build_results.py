from PIL import Image, ImageDraw, ImageFont
import glob

CAMP="runs/campaign_agentic_v2"
NAVY=(31,58,95); INK=(22,34,46); GRAY=(80,95,110)

def font(sz,bold=True):
    try: return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",sz)
    except: return ImageFont.load_default()

def bev_crop(sid,frac):
    paths=sorted(glob.glob(f"{CAMP}/{sid}/frames/frame_*.png"))
    im=Image.open(paths[int(len(paths)*frac)]).convert("RGB")
    w,h=im.size
    return im.crop((w//2+2,0,w,h))

# Canvas matches the slide figure area ratio (~2.6) so it fills it with no wasted space.
W,H=2400,900
c=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(c)

d.text((W//2,90),"All 23 scenarios correct, across 4 CARLA towns",font=font(96),fill=NAVY,anchor="mm")

thumbs=[("s01_t04_highway_safe_pass",0.60,"Highway"),
        ("s13_t01_rural_two_lane_pass",0.62,"Rural two-lane"),
        ("s09_t03_urban_pass",0.62,"Urban"),
        ("s22_t04_highway_heavy_traffic_pass",0.60,"Heavy traffic"),
        ("s23_t04_highway_truck_pass",0.60,"Truck lead")]
m=len(thumbs); side=64; gap=26
tw=(W-2*side-(m-1)*gap)//m          # ~433
th=int(tw*0.82)                     # taller thumbnails to fill height
y0=230
for i,(sid,frac,cap) in enumerate(thumbs):
    x=side+i*(tw+gap)
    try: im=bev_crop(sid,frac).resize((tw,th))
    except Exception: im=Image.new("RGB",(tw,th),(40,40,40))
    c.paste(im,(x,y0))
    d.rectangle([x,y0,x+tw,y0+th],outline=NAVY,width=4)
    d.text((x+tw//2,y0+th+50),cap,font=font(56),fill=INK,anchor="mm")

d.text((W//2,H-70),"Every decision made by the live agent: pass, wait, or yield-then-pass, gated by the critic.",
       font=font(56),fill=GRAY,anchor="mm")

c.save("presentation_assets/fig_results.png")
print("saved results",c.size,"thumb",tw,"x",th)
