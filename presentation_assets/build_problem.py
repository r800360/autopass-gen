import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

NAVY="#1F3A5F"; AMBER="#E08A2B"; RED="#D2433A"; INK="#16222E"; ROAD="#6E7780"; BLUE="#2477C8"

fig,ax=plt.subplots(figsize=(10.8,4.05),dpi=210)
ax.set_xlim(0,112); ax.set_ylim(0,42); ax.axis("off")

ax.add_patch(plt.Rectangle((4,9),104,22,fc=ROAD,ec="none",zorder=1))
for yy in (9,31):
    ax.plot([4,108],[yy,yy],color="white",lw=2.4,zorder=2)
for x0 in range(6,108,7):
    ax.plot([x0,x0+4],[20,20],color="white",lw=2.2,zorder=2)

def car(x,y,w,h,fc,label):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.2,rounding_size=1.6",fc=fc,ec="white",lw=1.6,zorder=4))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",color="white",fontsize=12.5,fontweight="bold",zorder=5)

car(70,11.5,12,6.5,RED,"LEAD")
car(33,11.5,12,6.5,BLUE,"EGO")
car(19,22.0,12,6.5,"#33414E","FAST")
ax.text(76,8.0,"slow",ha="center",color=RED,fontsize=12.5,fontweight="bold")
ax.add_patch(FancyArrowPatch((46,14.7),(58,14.7),arrowstyle="-|>",mutation_scale=16,lw=2.2,color="white",zorder=5))
ax.add_patch(FancyArrowPatch((32,25.2),(54,25.2),arrowstyle="-|>",mutation_scale=20,lw=3.0,color="#FFD24A",zorder=5))
ax.text(62,25.4,"closing fast",ha="left",va="center",color="#E0A21A",fontsize=12.5,fontweight="bold",zorder=5)

ax.add_patch(Circle((106,38.6),3.4,fc=AMBER,ec="none",zorder=4))
ax.text(106,38.6,"!",ha="center",va="center",color="white",fontsize=16,fontweight="bold",zorder=5)
ax.text(101.5,38.6,"deadline",ha="right",va="center",color=INK,fontsize=12.5,fontweight="bold",zorder=5)

ax.text(54,38.6,"Overtake now, or wait for a clear lane?",ha="center",va="center",color=NAVY,fontsize=18,fontweight="bold")
ax.text(56,3.7,"Safe only if the ego reads the REAR and ONCOMING lanes first,  not from a pre-planned trajectory.",
        ha="center",va="center",color=INK,fontsize=13.5)

plt.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01)
plt.savefig("presentation_assets/fig_problem.png",dpi=210,bbox_inches="tight",pad_inches=0.06,facecolor="white")
print("saved problem")
