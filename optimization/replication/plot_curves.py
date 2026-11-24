"""Make the appendix graph from integer checkpoint errors on paired teachers."""
import csv
import math
from pathlib import Path
import os

import numpy as np
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

root = Path(__file__).resolve().parent
files = sorted((root / "results").glob("metrics_*.csv"))
assert len(files) == 16, files
errors = {}
for file in files:
    for row in csv.DictReader(file.open(encoding="utf-8-sig", newline="")):
        key = (float(row["rate"]), row["arm"], int(row["labels"]), int(row["seed"]))
        assert key not in errors
        errors[key] = 100 * int(row["errors"]) / int(row["test"])
seeds = sorted({s for _, _, _, s in errors})
assert seeds == list(range(731901, 731917))
groups = ("clean", "noisy")
rates = (.003, .020)
budgets = sorted(set.intersection(*({b for r, a, b, s in errors if (r, a, s) == (rate, arm, seed)}
                                    for rate in rates for arm in ("final", *groups) for seed in seeds)))
assert budgets[-1] == 262144 and 131072 in budgets and len(budgets) >= 10
rng = np.random.default_rng(7319011)
resample = rng.integers(0, len(seeds), (50000, len(seeds)))
curves = {}
for rate in rates:
    for arm in groups:
        stats = []
        for budget in budgets:
            gaps = np.array([errors[rate,arm,budget,s] - errors[rate,"final",budget,s] for s in seeds])
            lo, hi = np.percentile(gaps[resample].mean(axis=1), [2.5,97.5])
            stats.append((gaps.mean(), lo, hi))
        curves[rate,arm] = np.array(stats)
ymin = min(-10, math.floor((min(vals[:,1].min() for vals in curves.values())-1)/10)*10)
ymax = max(10, math.ceil((max(vals[:,2].max() for vals in curves.values())+1)/10)*10)
xmin,xmax = map(math.log2,(budgets[0],budgets[-1]))
for name, font in [("TimesNR","times.ttf"), ("TimesNR-Bold","timesbd.ttf")]:
    path = Path(os.environ.get("TIMES_FONT_DIR", "C:/Windows/Fonts")) / font
    if path.is_file():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    else:
        pdfmetrics.registerFont(pdfmetrics.Font(name, "Times-Bold" if name.endswith("-Bold") else "Times-Roman", "WinAnsiEncoding"))
out = root / "results/confirmation_learning_curves.pdf"
canvas = Canvas(str(out), pagesize=(504,210), invariant=True)
canvas.setTitle("Matched-label error differences across training")
canvas.setAuthor("")
ink = HexColor("#20262C")
grey = HexColor("#D1D5D7")
colors = {"clean":HexColor("#1B5689"), "noisy":HexColor("#AB5432")}


def txt(x, y, s, size=10, bold=False, align="left"):
    canvas.setFillColor(ink)
    canvas.setFont("TimesNR-Bold" if bold else "TimesNR", size)
    getattr(canvas, {"left":"drawString","center":"drawCentredString","right":"drawRightString"}[align])(x,y,s)


def line(x0, y0, x1, y1, color=ink, width=.65):
    canvas.setLineWidth(width)
    canvas.setStrokeColor(color)
    canvas.line(x0,y0,x1,y1)


bottom, top = 37, 172
fy = lambda v: bottom + (v - ymin) / (ymax - ymin) * (top-bottom)
for j, rate in enumerate(rates):
    x0,x1 = [(43,246),(285,488)][j]
    def fx(b):
        return x0 + (math.log2(b)-xmin)/(xmax-xmin)*(x1-x0)
    txt((x0+x1)/2,181,"Adam rate %.3f"%rate,11,True,"center")
    for y in range(ymin,ymax+1,10):
        line(x0,fy(y),x1,fy(y),grey,.42)
        txt(x0-5,fy(y)-3,str(y).replace("-","\u2212"),8.8,align="right")
    for arm in groups:
        arr=curves[rate,arm]
        band=canvas.beginPath()
        band.moveTo(fx(budgets[0]), fy(arr[0,1]))
        for b,(_,lo,hi) in zip(budgets[1:],arr[1:]):
            band.lineTo(fx(b),fy(lo))
        for b,(_,lo,hi) in zip(reversed(budgets),arr[::-1]):
            band.lineTo(fx(b),fy(hi))
        band.close()
        canvas.saveState()
        canvas.setFillColor(colors[arm])
        canvas.setFillAlpha(.14)
        canvas.drawPath(band,stroke=0,fill=1)
        canvas.restoreState()
    canvas.setDash(2,2)
    line(x0,fy(0),x1,fy(0),ink,.65)
    canvas.setDash()
    for arm in groups:
        arr=curves[rate,arm]
        curve=canvas.beginPath()
        for i,(b,(mean,_,_)) in enumerate(zip(budgets,arr)):
            (curve.moveTo if i==0 else curve.lineTo)(fx(b),fy(mean))
        canvas.setLineWidth(1.35)
        canvas.setStrokeColor(colors[arm])
        if arm=="noisy":
            canvas.setDash(4,2)
        canvas.drawPath(curve)
        canvas.setDash()
        canvas.setFillColor(colors[arm])
        for b,(mean,_,_) in zip(budgets,arr):
            if arm=="clean":
                canvas.circle(fx(b),fy(mean),1.6,stroke=0,fill=1)
            else:
                canvas.rect(fx(b)-1.4,fy(mean)-1.4,2.8,2.8,stroke=0,fill=1)
    for b in [2048,8192,32768,131072,262144]:
        line(fx(b),bottom,fx(b),bottom-2.5)
        txt(fx(b),bottom-13,str(b//1024),9,align="center")
    line(x0,bottom,x1,bottom)
    line(x0,bottom,x0,top)
canvas.saveState()
canvas.translate(11,(bottom+top)/2)
canvas.rotate(90)
txt(0,0,"Error difference (percentage points)",10,align="center")
canvas.restoreState()
txt(264,5,"Total training labels (units of 1,024, log scale)",10.5,align="center")
for x,arm,s in [(107,"clean","Clean first sign"),(254,"noisy","10% noise in first sign")]:
    if arm=="noisy": canvas.setDash(4,2)
    line(x,201,x+19,201,colors[arm],1.35)
    canvas.setDash()
    txt(x+24,197.5,s,10)
canvas.showPage()
canvas.save()
print(len(files), len(budgets), budgets, out)

