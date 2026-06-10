import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

NAVY="#1F3A5F"; TEAL="#1390A6"; AMBER="#E08A2B"; GREEN="#2BA84A"; RED="#D2433A"
GRAY="#5B6B7B"; LNAVY="#EAF1F8"; INK="#16222E"

fig,ax=plt.subplots(figsize=(11.0,4.6),dpi=210)
ax.set_xlim(0,112); ax.set_ylim(0,56); ax.axis("off")

def box(x,y,w,h,title,sub,fc,tc="white",fs=15,subfs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.6,rounding_size=2.2",fc=fc,ec="none",zorder=3))
    ax.text(x+w/2,y+h*0.63,title,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold",zorder=4)
    if sub: ax.text(x+w/2,y+h*0.26,sub,ha="center",va="center",color=tc,fontsize=subfs,zorder=4)

def arrow(p1,p2,color=INK,lw=2.6,rad=0.0):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=20,lw=lw,color=color,
                 connectionstyle=f"arc3,rad={rad}",zorder=2))

y=30; h=11
box(2,y,18,h,"PERCEPTION","front + rear\nseg + depth → gaps",TEAL)
box(24,y,18,h,"PLANNER  (LLM)","chooses the\nnext tool",NAVY)
box(46,y,18,h,"CRITIC","deterministic\nsafety check",AMBER)
box(72,y,18,h,"CONTROLLER","waypoint lane-keep\n+ lane change",GREEN)
box(94,y,16,h,"CARLA","execute\n+ observe",GRAY)

arrow((20,y+h/2),(24,y+h/2))
arrow((42,y+h/2),(46,y+h/2))
arrow((64,y+h/2),(72,y+h/2),color=GREEN)
ax.text(68,y+h/2+2.6,"safe",ha="center",color=GREEN,fontsize=13,fontweight="bold")
arrow((90,y+h/2),(94,y+h/2),color=GREEN)

# reject path: critic -> planner (clean arc, below the boxes)
arrow((50,y),(34,y),color=RED,rad=-0.42,lw=2.6)
ax.text(42,y-6.0,"unsafe:  re-sense / re-plan",ha="center",color=RED,fontsize=12.5,fontweight="bold")

# closed-loop feedback: clean orthogonal route along the top
gy=48
ax.add_line(Line2D([102,102],[y+h,gy],color=GRAY,lw=2.6,zorder=2))
ax.add_line(Line2D([102,11],[gy,gy],color=GRAY,lw=2.6,zorder=2))
ax.add_patch(FancyArrowPatch((11,gy),(11,y+h),arrowstyle="-|>",mutation_scale=20,lw=2.6,color=GRAY,zorder=2))
ax.text(56,gy+2.4,"closed loop:  observe  →  update belief",ha="center",color=GRAY,fontsize=12.5,fontweight="bold")

# shared DSL bar
dx,dw=24,66; dy,dh=13,8.5
ax.add_patch(FancyBboxPatch((dx,dy),dw,dh,boxstyle="round,pad=0.6,rounding_size=2.2",fc=LNAVY,ec=NAVY,lw=1.8,zorder=3))
ax.text(dx+dw/2,dy+dh*0.64,"SHARED  DSL  (mutable)",ha="center",va="center",color=NAVY,fontsize=13.5,fontweight="bold",zorder=4)
ax.text(dx+dw/2,dy+dh*0.25,"belief  +  memory (denials, tool history)  +  plan",ha="center",va="center",color=INK,fontsize=10.5,zorder=4)
for cx in (33,55,81):
    ax.add_patch(FancyArrowPatch((cx,dy+dh),(cx,y),arrowstyle="<|-|>",mutation_scale=13,lw=1.8,color=NAVY,zorder=2))

# tool palette
tools=["sense_front","sense_rear","sense_passing_lane","check_corridor","propose_pass","hold"]
ax.text(2,4.2,"Tool palette:",color=INK,fontsize=11.5,fontweight="bold",va="center")
tx=15.5
for t in tools:
    w=len(t)*0.80+2.6
    ax.add_patch(FancyBboxPatch((tx,2.0),w,4.4,boxstyle="round,pad=0.2,rounding_size=1.6",fc="white",ec=NAVY,lw=1.4,zorder=3))
    ax.text(tx+w/2,4.2,t,ha="center",va="center",color=NAVY,fontsize=9.5,zorder=4)
    tx+=w+1.5

plt.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01)
plt.savefig("presentation_assets/fig_architecture.png",dpi=210,bbox_inches="tight",pad_inches=0.08,facecolor="white")
print("saved arch")
