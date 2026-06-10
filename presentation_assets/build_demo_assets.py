from PIL import Image, ImageDraw, ImageFont
def font(sz,bold=True):
    try: return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",sz)
    except: return ImageFont.load_default()
W,Hh=2576,720
NAVY=(15,27,44); TEAL=(70,205,205); WHITE=(245,248,252); GRAY=(175,190,205)

cards=[("presentation_assets/_cardA.png","Scenario 1  ·  Highway","Yield, then overtake",
        "A fast car is closing in the passing lane"),
       ("presentation_assets/_cardB.png","Scenario 2  ·  Rural two-lane","Wait for oncoming, then overtake",
        "Oncoming traffic in the only passing lane")]
for path,tag,title,sub in cards:
    im=Image.new("RGB",(W,Hh),NAVY); d=ImageDraw.Draw(im)
    # d.text((W//2,Hh*0.27),tag,font=font(66),fill=TEAL,anchor="mm")
    # d.text((W//2,Hh*0.50),title,font=font(150),fill=WHITE,anchor="mm")
    # d.text((W//2,Hh*0.74),sub,font=font(82,bold=False),fill=GRAY,anchor="mm")
    d.text((W//2,Hh*0.25),tag,font=font(82),fill=TEAL,anchor="mm")
    d.text((W//2,Hh*0.50),title,font=font(165),fill=WHITE,anchor="mm")
    d.text((W//2,Hh*0.76),sub,font=font(98,bold=False),fill=GRAY,anchor="mm")
    im.save(path); print("card",path)

# translucent lower-third banners (RGBA) overlaid during each clip
banners=[("presentation_assets/_bannerA.png","Holds for the fast car, then overtakes"),
         ("presentation_assets/_bannerB.png","Waits for oncoming, then overtakes")]
# BH=132
# for path,text in banners:
#     b=Image.new("RGBA",(W,BH),(10,18,30,180)); d=ImageDraw.Draw(b)
#     d.text((W//2,BH//2),text,font=font(72),fill=(255,255,255,255),anchor="mm")
#     b.save(path); print("banner",path)
BH = 190
for path, text in banners:
    b = Image.new("RGBA", (W, BH), (10, 18, 30, 215))
    d = ImageDraw.Draw(b)
    d.text((W//2, BH//2), text, font=font(110), fill=(255,255,255,255), anchor="mm")
    b.save(path)
    print("banner", path)