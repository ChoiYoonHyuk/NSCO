"""Render the two Gaussian checks from released summaries, with embedded Times New Roman."""
import csv
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path(__file__).resolve().parent
font_dir=Path(os.environ.get("PAPER_TIMES_DIR",r"C:\Windows\Fonts"))
for file in ("times.ttf","timesbd.ttf","timesi.ttf","timesbi.ttf"):
    if (font_dir/file).is_file():
        font_manager.fontManager.addfont(font_dir/file)
family="Times New Roman" if (font_dir/"times.ttf").is_file() else "DejaVu Serif"
plt.rcParams.update({"font.family":family,"font.size":9,"pdf.fonttype":42,
                     "axes.spines.right":False,"axes.spines.top":False,"axes.titlesize":10})

def rows(name):
    with (root/"inputs"/name).open() as f:
        return list(csv.DictReader(f))

train=rows("gaussian/results/matched_curves.csv")
overlap=rows("overlap/curves.csv")
fig,(left,right)=plt.subplots(1,2,figsize=(7.05,2.75))
for opt,style in (("SGD","-"),("Adam","--")):
    for task,color,marker in (("A","#196384","o"),("B","#a84326","s")):
        arm=sorted([r for r in train if (r["optimizer"],r["readout"])==(opt,task)],key=lambda r:int(r["n"]))
        assert len(arm)==8
        x=np.log2([int(r["n"]) for r in arm])
        left.plot(x,[100*float(r["mean_error"]) for r in arm],color=color,marker=marker,
                  linestyle=style,lw=1.35,ms=3,label=f"{opt}, {task}")
left.set(title="(a) Same global stability, different local mass",ylabel="Clean answer error (%)",
         xlabel="Training examples",xticks=[10,12,14,16,17],xticklabels=["1,024","4,096","16,384","65,536","131,072"])
for i,lab in enumerate(left.get_xticklabels()):
    lab.set_rotation(15)
    lab.set_ha("right")
colors=["#353938","#aa502c","#12658a"]
for j,(key,name,mark) in enumerate([("oracle","True vector","o"),("short_given_direction","Shortened vector","s"),
                                    ("joint_grid","Calibrated vector","^")]):
    arm=sorted([r for r in overlap if r["arm"]==key and int(r["n"])==2**21],key=lambda r:float(r["rho"]))
    assert len(arm)==3
    y=np.array([100*float(r["clean_error"]) for r in arm])
    lo=np.array([100*float(r["clean_error_ci_lo"]) for r in arm])
    hi=np.array([100*float(r["clean_error_ci_hi"]) for r in arm])
    right.errorbar(np.arange(3)+(j-1)*.15,y,yerr=np.array([y-lo,hi-y]),fmt=mark,ms=4,
                   capsize=2,lw=1.1,color=colors[j],label=name)
right.set(title="(b) Learning the overlap correction",xlabel="True overlap magnitude",
          ylabel="Clean parity error (%)",xticks=[0,1,2],xticklabels=["0.5","0.9 (off grid)","1.0"],xlim=(-.35,2.35))
for ax in (left,right):
    ax.set_ylim(0,52)
    ax.set_yticks([0,10,20,30,40,50])
    ax.grid(axis="y",linewidth=.5,color="#dfdfdf")
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5)
    ax.legend(loc="upper left",ncol=2 if ax==left else 1,frameon=False,fontsize=7.6,
              handlelength=1.5,borderpad=.15,labelspacing=.17,columnspacing=.65)
fig.subplots_adjust(left=.077,right=.984,bottom=.25,top=.88,wspace=.31)
folder=root/"output"
folder.mkdir(parents=True,exist_ok=True)
fig.savefig(folder/"reviewer_checks.pdf")
fig.savefig(folder/"reviewer_checks.png",dpi=160)
