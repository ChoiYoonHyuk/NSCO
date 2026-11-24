"""Prespecified paired contrasts, exact sign flips, and annotation accounting."""
import csv
import itertools
import json
import math
import statistics as st
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
P=json.loads((ROOT/"protocol.json").read_text())
seeds=P["seeds"];grid=P["n"]


def load(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def write(name,rows):
    with (OUT/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def ci(v):
    radius={5:2.776445105,12:2.20098516}[len(v)]*st.stdev(v)/math.sqrt(len(v))
    return [st.mean(v)-radius,st.mean(v)+radius]


def p_exact(v):
    observed=abs(sum(v))
    values=(abs(sum(k*x for k,x in zip(signs,v))) for signs in itertools.product([-1,1],repeat=len(v)))
    return sum(x>=observed-1e-12 for x in values)/2**len(v)


def restricted_median(v):
    finite=sorted(x for x in v if x is not None)
    lo=(len(v)-1)//2;hi=len(v)//2
    return (finite[lo]+finite[hi])/2 if len(finite)>hi else None


def current():
    lookup={(int(r["seed"]),r["optimizer"],r["readout"],int(r["n"])):r for r in load(OUT/"measurements.csv") if r["deployment"]=="polar"}
    primary=[]; paired=[];threshold=[]; cells=[]; curves=[]
    for opt in ["SGD","Adam"]:
        bytask={}
        for task in ["A","B"]:
            hits={}; summary={}
            for n in grid:
                vals=[float(lookup[s,opt,task,n]["error"]) for s in seeds]
                curves.append(dict(optimizer=opt,readout=task,n=n,mean_error=st.mean(vals),
                                   lo=ci(vals)[0],hi=ci(vals)[1]))
            for metric,target in [("error",.1),("frame",.35)]:
                first=[next((n for n in grid if float(lookup[s,opt,task,n][metric])<=target),None) for s in seeds]
                for s,h in zip(seeds,first):
                    threshold.append(dict(seed=s,optimizer=opt,readout=task,target=metric,grid_crossing=h,
                                          censored=int(h is None),maximum=grid[-1]))
                summary[metric+"_successes"]=sum(h is not None for h in first)
                summary[metric+"_median"]=restricted_median(first)
            av=[st.mean(float(lookup[s,opt,task,n]["error"]) for n in grid) for s in seeds]
            bytask[task]=av
            cells.append(dict(optimizer=opt,readout=task,average_grid_error=st.mean(av),
                              endpoint_error=st.mean(float(lookup[s,opt,task,grid[-1]]["error"]) for s in seeds),
                              endpoint_frame=st.mean(float(lookup[s,opt,task,grid[-1]]["frame"]) for s in seeds),
                              endpoint_path=st.mean(float(lookup[s,opt,task,grid[-1]]["complete_path"]) for s in seeds),**summary))
        contrasts=[b-a for a,b in zip(bytask["A"],bytask["B"])]
        for k,(s,a,b,d) in enumerate(zip(seeds,bytask["A"],bytask["B"],contrasts)):
            paired.append(dict(seed=s,optimizer=opt,integrated_A=a,integrated_B=b,B_minus_A=d))
        primary.append(dict(optimizer=opt,B_minus_A=st.mean(contrasts),t95_low=ci(contrasts)[0],
                            t95_high=ci(contrasts)[1],p=p_exact(contrasts),positive_seeds=sum(d>0 for d in contrasts)))
    p0,p1=sorted(primary,key=lambda x:x["p"])
    p0["holm_p"]=min(1,2*p0["p"]);p1["holm_p"]=max(p0["holm_p"],p1["p"])
    write("primary.csv",primary);write("matched_seed_contrasts.csv",paired);write("matched_thresholds.csv",threshold)
    write("matched_cells.csv",cells);write("matched_curves.csv",curves)
    gradients=load(OUT/"gradients.csv")
    return {"primary":primary,"cells":cells,"snr":{x:st.mean(float(r["snr_squared"]) for r in gradients if r["readout"]==x) for x in ["A","B"]}}


def accounting():
    raw=load(ROOT/"prior_cost_inputs.csv")
    r={(int(x["seed"]),x["readout"],x["optimizer"],x["supervision"],int(x["n"])):float(x["error"]) for x in raw}
    oldseeds=sorted({k[0] for k in r})
    budgets=grid[1:]
    pair=[]; thresholds=[]; end=[]
    for t,opt in itertools.product(["First","Last","Parity","MajProd"],["sgd","adam"]):
        for sup in ["final","value","index"]:
            mult=1 if sup=="final" else 2
            for s in oldseeds:
                first=next((n for n in grid if r[s,t,opt,sup,n]<=.10),None)
                thresholds.append(dict(seed=s,readout=t,optimizer=opt,supervision=sup,
                    first_success_training_records=first,first_success_bit_observations=None if first is None else mult*first,
                    standalone_development_bits=131072*mult+4096,maximum_observed_training_bits=131072*mult,
                    right_censored=int(first is None)))
            for bit in budgets:
                n=bit//mult
                de=[]
                for s in oldseeds:
                    e=r[s,t,opt,sup,n]; base=r[s,t,opt,"final",bit]
                    de.append(e-base)
                    pair.append(dict(seed=s,readout=t,optimizer=opt,supervision=sup,bit_budget=bit,
                             noisy_answer_labels=n,clean_aux_labels=n*(mult-1),error=e,increment_vs_final=e-base))
                if bit==budgets[-1]:
                    e=[r[s,t,opt,sup,n] for s in oldseeds]
                    end.append(dict(readout=t,optimizer=opt,supervision=sup,bit_budget=bit,
                                    mean_error=st.mean(e),delta_vs_final=st.mean(de),lo=ci(de)[0],hi=ci(de)[1]))
    write("prior_matched_bit_curves.csv",pair);write("prior_bit_thresholds.csv",thresholds)
    write("prior_equal_bit_endpoints.csv",end)
    assert 131072+4096+131072 < 2*131072+4096+2*1024
    return end


if __name__=="__main__":
    summary=current()
    accounting()
    (OUT/"headline.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
