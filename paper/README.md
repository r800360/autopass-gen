# AutoPass-Gen paper (reframed)

## Build

```powershell
cd paper
# Copy diagram from presentation assets (or your passing-diagram.png)
copy ..\presentation\assets\passing-diagram.png .
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Requires `cvpr.sty` on your TeX path (same as course template).

## Changes from prior draft

- **Framing:** safe-but-slow AVs → **greedy urgency-aware passing** + multi-agent LLM/code split
- **Removed:** TikZ closed-loop Fig. 2 (generator–mutator loop)
- **Kept:** Xinwei's passing pipeline diagram as the single architecture figure
- **Aligned with codebase:** LangGraph, seg/depth tools, mock LLMs, CARLA watch, `plans_are_same` replan check
- **TA clarity:** where scenes and numbers come from; code vs LLM table; honest CARLA scope
