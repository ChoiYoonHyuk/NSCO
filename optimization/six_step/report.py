"""Summarize grid crossings at seed level and draw paired-design learning curves."""
import csv
import itertools
import math
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
fields=["readout","supervision","optimizer","init","deployment"]
tasks=["First","Last","Parity","MajProd"]
supervisions=["final","value","index"]
grid=[2**i for i in range(10,18)]


def read(name):
    with name.open() as f:
        return list(csv.DictReader(f))


def save(name,rows):
    with (OUT/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def key(row):
    return tuple(row[k] for k in fields)


def interval(x):
    delta=2.776445105*stdev(x)/math.sqrt(len(x))
    return mean(x)-delta,mean(x)+delta


def report():
    rows=[r for f in sorted(OUT.glob("rows_*.csv")) for r in read(f)]
    seeds=sorted({r["seed"] for r in rows})
    cells=sorted({key(r) for r in rows})
    lookup={(key(r),r["seed"],int(r["n"])):r for r in rows}
    summary, crossings, curves=[],[],[]
    for cell in cells:
        info=dict(zip(fields,cell))
        hits=[]
        for seed in seeds:
            record=[n for n in grid if float(lookup[cell,seed,n]["error"])<=.1]
            if record:
                hits.append(record[0])
            crossings.append(dict(**info,seed=seed,first_grid_threshold=record[0] if record else None,
                                  right_censored=int(not record),last_grid=grid[-1]))
        metrics=["error","path","frame_error","projector_error"]+[f"hop{h}" for h in range(1,7)]
        endpoint={}
        for n in grid:
            curve={}
            for metric in metrics:
                values=[lookup[cell,seed,n][metric] for seed in seeds]
                v=[float(x) for x in values if x]
                curve[metric]=mean(v) if v else None
                if metric=="error":
                    curve.update(error_low=interval(v)[0],error_high=interval(v)[1])
            curves.append(dict(**info,n=n,**curve))
            endpoint=curve
        summary.append(dict(**info,**endpoint,threshold_successes=len(hits),seed_count=len(seeds),
                            median_reached=median(hits) if hits else None,
                            median_with_censor=(sorted(hits)[len(seeds)//2]
                                                if len(hits)>len(seeds)//2 else None)))
    save("endpoints.csv",summary)
    save("thresholds.csv",crossings)
    save("curves.csv",curves)
    gradients=[r for f in sorted(OUT.glob("gradient_*.csv")) for r in read(f)]
    gs=[]
    for coord, grp in itertools.groupby(sorted(gradients,key=lambda r:tuple(r[k] for k in fields[:4])),
                                         key=lambda r:tuple(r[k] for k in fields[:4])):
        group=list(grp)
        out={}
        for k in ["mean_norm","noise_rms","unbiased_signal_sq","snr_squared"]:
            v=[float(r[k]) for r in group if r[k]]
            out[k]=mean(v) if v else None
        gs.append(dict(zip(fields[:4],coord),**out))
    save("gradient_summary.csv",gs)
    differences=[]
    for opt,init in itertools.product(["sgd","adam"],["zero","small"]):
        for task in tasks:
            for s1,s0 in [("index","final"),("value","final"),("index","value")]:
                a=(task,s1,opt,init,"polar")
                b=(task,s0,opt,init,"polar")
                diff=[float(lookup[a,seed,grid[-1]]["error"])-float(lookup[b,seed,grid[-1]]["error"]) for seed in seeds]
                low,high=interval(diff)
                differences.append(dict(readout=task,optimizer=opt,init=init,contrast=f"{s1} - {s0}",
                                        mean_difference=mean(diff),t95_lower=low,t95_upper=high))
    save("paired_differences.csv",differences)
    return curves


def plot(curves):
    W,H=522,323
    pdf=canvas.Canvas(str(ROOT/"gaussian_curves.pdf"),pagesize=(W,H))
    pdf.setTitle("Gaussian retrieval learning curves")
    colors=["#136B64","#BE4C2A","#3E5AA4","#75518A"]
    fmt={"final":"Final answer","value":"Final + first value","index":"Final + first index"}
    def label(x,y,s,size=9,align="left",bold=False):
        pdf.setFillColor(HexColor("#202327"));pdf.setFont("Helvetica-Bold" if bold else "Helvetica",size)
        {"left":pdf.drawString,"center":pdf.drawCentredString,"right":pdf.drawRightString}[align](x,y,s)
    for i,t in enumerate(tasks):
        x=104+i*87
        pdf.setStrokeColor(HexColor(colors[i]));pdf.setLineWidth(1.8)
        pdf.setDash([4,2] if i==1 else [1.4,1.4] if i==3 else [])
        pdf.line(x,305,x+15,305);pdf.setDash([])
        label(x+19,301.8,t)
    for y_id,opt in enumerate(["sgd","adam"]):
        for x_id,sup in enumerate(supervisions):
            x0=45+x_id*158; y0=171-y_id*129; w=137;h=93
            if not y_id:
                label(x0+w/2,y0+h+12,fmt[sup],10,"center",True)
            if x_id==0:
                label(x0,y0+h+3,opt.upper() if opt=="sgd" else "Adam",8,bold=True)
            for y in [0,.1,.2,.3,.4,.5]:
                yy=y0+h*y/.52
                pdf.setStrokeColor(HexColor("#9A9DA3" if y==.1 else "#E0E2E5"))
                pdf.setLineWidth(.5);pdf.setDash([2,2] if y==.1 else [])
                pdf.line(x0,yy,x0+w,yy);pdf.setDash([])
                if x_id==0:
                    label(x0-4,yy-2.5,str(round(y*100)),7,"right")
            for j,n in enumerate(grid):
                xx=x0+w*j/(len(grid)-1)
                if y_id and j in [0,2,4,6,7]:
                    label(xx,y0-11,str(n//1024)+"k",7,"center")
            for i,t in enumerate(tasks):
                data=sorted([r for r in curves if tuple(r[k] for k in fields)==(t,sup,opt,"small","polar")],key=lambda r:r["n"])
                pdf.setStrokeColor(HexColor(colors[i]));pdf.setLineWidth(1.4)
                if i==1:pdf.setDash([4,2])
                if i==3:pdf.setDash([1.4,1.4])
                path=pdf.beginPath()
                for j,r in enumerate(data):
                    xx=x0+w*j/7;yy=y0+h*r["error"]/.52
                    (path.moveTo if j==0 else path.lineTo)(xx,yy)
                pdf.drawPath(path);pdf.setDash([])
    label(W/2,12,"Training examples (1k = 1,024)",9,"center")
    pdf.saveState();pdf.translate(11,H/2);pdf.rotate(90);label(0,0,"Clean hard prediction error (%)",9,"center");pdf.restoreState()
    pdf.showPage();pdf.save()


if __name__=="__main__":
    plot(report())
