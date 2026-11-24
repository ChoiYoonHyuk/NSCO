"""Audit the streamed risk trajectories and render the new two-panel paper figure."""
from pathlib import Path
from collections import defaultdict
import csv
import os
import hashlib
import json
import math
import numpy as np
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor, white
from pypdf import PdfReader

root=Path(__file__).resolve().parent
from study import P as pp

sourcehash = hashlib.sha256((root / "study.py").read_bytes()).hexdigest()
seeds, ks = pp["seeds"], pp["depths"]
trajectories = defaultdict(list)
for seed in seeds:
    results = root / "results"
    receipt = json.loads((results / f"{seed}.json").read_text())
    if receipt["sha256"] != sourcehash:
        raise ValueError(f"Source differs from the recorded run for seed {seed}")
    with (results / f"{seed}.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == receipt["rows"]
    for row in rows:
        assert int(row["seed"]) == seed
        key = (seed, int(row["k"]), row["family"], row["supervision"])
        trajectories[key].append((int(row["labels"]), float(row["clean_error"])))
expected={(k,f,m) for k in ks for f in ("parity","majority_product") for m in ("final","plus_C1")}
expected|={(k,"terminal","final") for k in ks}
assert set(trajectories)=={(s,*spec) for s in seeds for spec in expected}
cross={}
records=[]
for key,curve in trajectories.items():
    s,k,f,m=key
    counts,risk=np.array(curve).T
    assert np.all(np.diff(counts)>0) and np.all(np.isclose(risk*pp["test_n"],np.round(risk*pp["test_n"])))
    assert len(risk)==1 or np.all(risk[:-1]>pp["target"])
    labels=2**np.arange(pp["log2_budgets"][0],pp["log2_budgets"][0]+len(curve))*(2 if m=="plus_C1" else 1)
    assert np.array_equal(counts,labels)
    observed=risk[-1]<=pp["target"]
    if not observed: assert len(curve)==len(pp["log2_budgets"])
    state="right" if not observed else "left" if len(curve)==1 else "bracket"
    x=float(np.log2(counts[-1]))
    if state=="bracket":
        x-=1-math.log(risk[-2]/pp["target"])/math.log(risk[-2]/max(risk[-1],.5/pp["test_n"]))
    cross[key]=(x,state)
    records.append(dict(seed=s,k=k,family=f,supervision=m,last_tested_total_labels=int(counts[-1]),
                        clean_error_at_last_tested=risk[-1],log2_interpolated_crossing=x if state=="bracket" else "",
                        censoring=state))
with (root/"crossings.csv").open("w",newline="") as fp:
    w=csv.DictWriter(fp,fieldnames=records[0]);w.writeheader();w.writerows(records)
inds=np.random.default_rng(412003).integers(len(seeds),size=(10000,len(seeds)))
def estimates(x):
    x=np.asarray(x,dtype=float);samps=x[inds].mean(axis=1)
    lo,hi=np.quantile(samps,[.025,.975])
    return [float(x.mean()),float(lo),float(hi)]
stats={};slopes={};reference={"parity":1.0,"majority_product":math.log2(32/13)/3,"terminal":0.0}
for f,m in sorted({(f,m) for _,f,m in expected}):
    akey=f+"/"+m
    data=np.array([[cross[s,k,f,m][0] for k in ks] for s in seeds])
    states=np.array([[cross[s,k,f,m][1] for k in ks] for s in seeds])
    stats[akey]={}
    for j,k in enumerate(ks):
        n=int(np.sum(states[:,j]=="bracket"))
        # For a right-censored seed store the conservative log-grid cap and
        # label the plotted point as a mean lower bound. Never fit that arm.
        if np.any(states[:,j]=="left"):
            raise ValueError("Left-censored arm requires explicit handling")
        stats[akey][k]=dict(num_bracketed=n,summary=estimates(data[:,j]),lower_bound=n<len(seeds))
    if np.all(states=="bracket"):
        ds=np.array(ks)-np.mean(ks)
        slopes[akey]=estimates(data@ds/(ds@ds))
summary={"source_sha256":sourcehash,"cap":max(pp["log2_budgets"]),"accuracy_target":pp["target"],
         "pointwise_paired_seed_bootstrap":10000,"reference_slopes":reference,
         "log2_total_label_cost":stats,"within_seed_slopes":slopes}
(root/"summary.json").write_text(json.dumps(summary,indent=2))

fontdir=Path(os.environ.get("TIMES_FONT_DIR", "C:/Windows/Fonts"))
has_fonts = all((fontdir/f).is_file() for f in ("times.ttf","timesbd.ttf"))
for name,file in [("TNR","times.ttf"),("TNR-B","timesbd.ttf")]:
    if has_fonts:
        pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
    else:
        face = "Times-Roman" if name == "TNR" else "Times-Bold"
        pdfmetrics.registerFont(pdfmetrics.Font(name, face, "WinAnsiEncoding"))
w,h=482,164
pdf=Canvas(str(root/"depth_supervision.pdf"),pagesize=(w,h),pageCompression=1)
pdf.setTitle("Clean prediction cost by readout and supervision")
ink="#182b3b";blue="#214d74";orange="#a54d29";green="#226c5c";grid="#d7dcde"
def txt(x,y,t,size,align="left",bold=False,col=ink):
    pdf.setFillColor(HexColor(col));pdf.setFont("TNR-B" if bold else "TNR",size)
    {"left":pdf.drawString,"center":pdf.drawCentredString,"right":pdf.drawRightString}[align](x,y,t)
def line(x1,y1,x2,y2,col,width=.65,dash=None):
    pdf.setStrokeColor(HexColor(col));pdf.setLineWidth(width);pdf.setDash(dash or [])
    pdf.line(x1,y1,x2,y2);pdf.setDash([])
def mark(x,y,style,color,sz=2.3):
    pdf.setStrokeColor(HexColor(color));pdf.setFillColor(HexColor(color))
    if style=="circle":pdf.circle(x,y,sz,stroke=1,fill=1)
    elif style=="square":pdf.rect(x-sz,y-sz,2*sz,2*sz,stroke=1,fill=1)
    else:
        path=pdf.beginPath();path.moveTo(x,y+1.23*sz);path.lineTo(x+sz,y-sz);path.lineTo(x-sz,y-sz);path.close()
        pdf.drawPath(path,stroke=1,fill=1)
display=[[("parity/final",blue,"circle","Parity"),("majority_product/final",orange,"square","Maj3 product"),("terminal/final",green,"tri","Terminal")],
         [("parity/final",blue,"circle","Final only"),("parity/plus_C1",orange,"square","Final + first value")]]
values=[a["summary"][j] for panel in display for key,*_ in panel for a in stats[key].values() for j in (1,2)]
ymin=math.floor((min(values)-.65)/2)*2;ymax=math.ceil((max(values)+.45)/2)*2
for idx,series in enumerate(display):
    x0,x1=(34,220) if idx==0 else (278,464)
    y0,y1=33,126
    X=lambda k:x0+9+(k-3)/6*(x1-x0-18)
    Y=lambda y:y0+(y-ymin)/(ymax-ymin)*(y1-y0)
    txt(x0-6,151,["(a) Final-label cost by readout","(b) Parity with intermediate supervision"][idx],10.1,bold=True)
    for tick in range(ymin,ymax+1,2):
        line(x0,Y(tick),x1,Y(tick),grid,.35)
        txt(x0-4,Y(tick)-2.7,str(tick),8.5,"right")
    line(x0,y0,x0,y1,ink);line(x0,y0,x1,y0,ink)
    for k in ks:
        line(X(k),y0,X(k),y0-2,ink)
        txt(X(k),y0-11,str(k),8.5,"center")
    txt((x0+x1)/2,9,"Depth k",9,"center")
    pdf.saveState();pdf.translate(x0-22,(y0+y1)/2);pdf.rotate(90)
    txt(0,0,"log₂ total training labels",9,"center");pdf.restoreState()
    last=x0
    for j,(key,color,shape,label) in enumerate(series):
        mark(last+2,138,shape,color,1.8);txt(last+7,135.5,label,8)
        last+=pdfmetrics.stringWidth(label,"TNR",8)+18
        arm=stats[key]
        ys=[arm[k]["summary"][0] for k in ks]
        if idx==0:
            slope=reference[key.split("/")[0]]
            line(X(3),Y(ys[0]),X(9),Y(ys[0]+6*slope),color,.7,[2.4,2.3])
        for k1,k2,a,b in zip(ks,ks[1:],ys,ys[1:]):
            line(X(k1),Y(a),X(k2),Y(b),color,1.35)
        for k in ks:
            value,lo,hi=arm[k]["summary"];x=X(k);y=Y(value)
            if not arm[k]["lower_bound"]:
                line(x,Y(lo),x,Y(hi),color,.85);line(x-2,Y(lo),x+2,Y(lo),color,.85);line(x-2,Y(hi),x+2,Y(hi),color,.85)
            else:
                line(x,y+2,x,y+8,color,1);line(x,y+8,x-2,y+5,color,1);line(x,y+8,x+2,y+5,color,1)
            mark(x,y,shape,color)
pdf.showPage();pdf.save()
page=PdfReader(root/"depth_supervision.pdf").pages[0]
fonts=[str(v.get_object()["/BaseFont"]) for v in page["/Resources"]["/Font"].values()]
if has_fonts:
    assert any("TimesNewRoman" in name for name in fonts),fonts
print(json.dumps(summary["within_seed_slopes"]))
print("all seeded crossings",[(t,int(sum(k[1]=="bracket" for k in cross.values())),len(cross)) for t in ["reports"]])
print("font names",fonts)
