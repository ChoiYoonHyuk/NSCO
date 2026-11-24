"""Build the paper's three-panel comparison from released numerical summaries."""
from pathlib import Path
import os
import csv
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

root=Path(__file__).resolve().parent
def rows(path):
    with (root/"inputs"/path).open() as f:
        return list(csv.DictReader(f))

gauss=rows("neural/results/curves.csv")
signed=json.loads((root/"inputs/hop/summary.json").read_text())["sign"]
alignment=rows("hop/shifted/results.csv")
depth=rows("overlap/cell_summary.csv")
colors=["#185a79","#bd512d","#705693","#408465"]
markers=["o","s","^","D"]
fontdir=Path(os.environ.get("PAPER_TIMES_DIR", r"C:\Windows\Fonts"))
for file in ("times.ttf","timesi.ttf","timesbd.ttf","timesbi.ttf"):
    if (fontdir/file).is_file():
        font_manager.fontManager.addfont(fontdir/file)
family="Times New Roman" if (fontdir/"times.ttf").is_file() else "DejaVu Serif"
plt.rcParams.update({"font.family":family,"font.size":8,"axes.titlesize":9,
                     "mathtext.fontset":"custom","mathtext.rm":family,
                     "mathtext.it":family+":italic","mathtext.bf":family+":bold",
                     "mathtext.cal":family+":italic","mathtext.sf":family,
                     "mathtext.tt":family,"mathtext.fallback":None,
                     "pdf.fonttype":42,"axes.spines.right":False,"axes.spines.top":False})
fig, axes=plt.subplots(1,3,figsize=(7.05,2.55))
for ax in axes:
    ax.grid(axis="y",alpha=.2,lw=.6)
    ax.tick_params(length=2,pad=2)
for j,(g,label) in enumerate([("First","First"),("Last","Last"),("Parity","Parity"),("MajProd","Majority product")]):
    sel=sorted((r for r in gauss if (r["optimizer"],r["init"],r["deployment"],r["supervision"],r["readout"])==
                ("sgd","small","polar","final",g)),key=lambda r:int(r["n"]))
    axes[0].plot([math.log2(int(r["n"])) for r in sel],[100*float(r["error"]) for r in sel],
                 lw=1.4,marker=markers[j],ms=2.6,color=colors[j],label=label)
budgets=[128,512,2048,8192]
for j,(kind,label) in enumerate([("successor","Successor key"),("same_position","Same position"),("Transformer","Transformer")]):
    if kind=="Transformer":
        row=next(r for r in signed if (r["model"],r["readout"])==("Transformer","parity"))
        ys=[row["curve_pct"][str(n)][0] for n in budgets]
    else:
        ys=[]
        for n in budgets:
            vals=[100*float(r["hard_error"]) for r in alignment if r["alignment"]==kind and r["readout"]=="parity" and int(r["n"])==n]
            assert len(vals)==5,sorted(set(r["alignment"] for r in alignment))
            ys.append(np.mean(vals))
    axes[1].plot(np.log2(budgets),ys,lw=1.4,marker=markers[j],ms=2.7,color=colors[j],label=label)
for ax,title,ticks in zip(axes[:2],["(a) Gaussian attention, SGD","(b) Three-hop token parity"],[[10,12,14,16,17],[7,9,11,13]]):
    ax.set(title=title,ylim=(-1.5,55),xticks=ticks,yticks=[0,10,25,50],xlabel=r"$\log_2$ training examples")
    ax.axhline(10,color=".6",ls=":",lw=.8)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.25),ncol=2,fontsize=6.1,frameon=False,
              handlelength=1.45,columnspacing=.9,labelspacing=.25,borderpad=.05)
axes[0].set_ylabel("Clean answer error (%)")
for j,rho in enumerate([0.,.5,1.]):
    sel=sorted((r for r in depth if int(r["D"])==16 and float(r["rho"])==rho and
                r["setting"]=="exact" and r["method"]=="sign_magnitude"),key=lambda r:int(r["k"]))
    assert len(sel)==4,sorted(set(r["method"] for r in depth))
    assert all(int(r["frame_success"])==6 for r in sel)
    axes[2].plot([int(r["k"])+[0,-.045,.045][j] for r in sel],[math.log2(float(r["frame_median_or_lower"])) for r in sel],
                 color=colors[j],ls=["-","--","-"][j],lw=1.4,marker=markers[j],ms=3,label=rf"$\rho={rho:g}$")
axes[2].set(title="(c) Gaussian frame recovery",xlabel="Retrieval depth", ylabel=r"$\log_2$ median label cost",
            xticks=[3,5,7,9],yticks=[12,14,16,18,20],ylim=(11.5,20))
axes[2].legend(loc="upper center",bbox_to_anchor=(.5,-.25),ncol=3,fontsize=6,frameon=False,
               handlelength=1.2,columnspacing=.55,borderpad=.05)
fig.subplots_adjust(left=.069,right=.986,bottom=.31,top=.90,wspace=.35)
folder=root/"output"
folder.mkdir(exist_ok=True,parents=True)
fig.savefig(folder/"followup_comparison.pdf")
fig.savefig(folder/"followup_comparison.png",dpi=170)
