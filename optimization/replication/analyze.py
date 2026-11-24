"""Aggregate all planned integer counts. No arm or teacher is excluded."""
import csv
import itertools
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
P=json.loads((ROOT/"plan.json").read_text())

def write(name,rows):
    with (OUT/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def collect():
    idx={}
    for s in P["seeds"]:
        expect={(r,a,n) for r,a,n in itertools.product(P["rates"],P["arms"],P["grid_examples"]) if n<=P["max_examples"][a]}
        with (OUT/f"metrics_{s}.csv").open() as f:
            for row in csv.DictReader(f):
                r,a,n=float(row["rate"]),row["arm"],int(row["n"])
                expect.remove((r,a,n))
                assert int(row["seed"])==s and int(row["depth"])==12
                assert int(row["test"])==P["test_examples"] and int(row["labels"])==n*(1+(a!="final"))
                assert 0<=int(row["errors"])<=int(row["test"]) and 0<=int(row["path_hits"])<=int(row["test"])
                idx[s,r,a,n]=int(row["errors"])
        assert not expect
    return idx

def signp(d):
    pos,neg=int((d>0).sum()),int((d<0).sum())
    n=pos+neg
    return min(1.,2*sum(math.comb(n,j) for j in range(min(pos,neg)+1))/2**n)

def main():
    idx=collect()
    seeds=P["seeds"]
    bootstrap=np.random.default_rng(7319011).integers(len(seeds),size=(50000,len(seeds)))
    arms,endpoints,pairs,crosses,byseed=[],[],[],[],[]
    for unit,b in [("labels",b) for b in P["label_caps"]]+[("examples",131072)]:
        errors={}
        for r,a in itertools.product(P["rates"],P["arms"]):
            cost=1+int(unit=="labels" and a!="final")
            n=b//cost
            e=np.array([idx[s,r,a,n] for s in seeds],dtype=float)*100/P["test_examples"]
            errors[r,a]=e
            arms.append(dict(unit=unit,budget=b,rate=r,arm=a,n=n,labels=n*(1+(a!="final")),mean_pct=e.mean(),median_pct=np.median(e)))
            first=[]
            for s in seeds:
                hit=next((t*cost for t in P["grid_examples"] if t<=n and idx[s,r,a,t]*10<=P["test_examples"]),math.inf)
                first.append(hit)
                byseed.append(dict(seed=s,unit=unit,budget=b,rate=r,arm=a,crossing=hit if math.isfinite(hit) else "",censored=int(not math.isfinite(hit))))
            first.sort()
            m=len(first)//2
            crosses.append(dict(unit=unit,budget=b,rate=r,arm=a,reaches=sum(math.isfinite(x) for x in first),teachers=len(first),middle_lower=first[m-1],middle_upper=first[m]))
        contrasts=[(r,a,r,"final") for r,a in itertools.product(P["rates"],["clean","noisy"])]
        contrasts += [(r,"noisy",r,"clean") for r in P["rates"]]+[(.003,a,.02,"final") for a in ["clean","noisy"]]
        for r,a,br,ba in contrasts:
            d=errors[r,a]-errors[br,ba]
            lo,hi=np.quantile(d[bootstrap].mean(axis=1),[.025,.975])
            endpoints.append(dict(unit=unit,budget=b,rate=r,arm=a,base_rate=br,base_arm=ba,arm_mean_pct=errors[r,a].mean(),base_mean_pct=errors[br,ba].mean(),delta_pp=d.mean(),ci95_low=lo,ci95_high=hi,better=int((d<0).sum()),ties=int((d==0).sum()),teachers=len(d),sign_p=signp(d)))
            for i,s in enumerate(seeds):
                pairs.append(dict(seed=s,unit=unit,budget=b,rate=r,arm=a,base_rate=br,base_arm=ba,arm_pct=errors[r,a][i],base_pct=errors[br,ba][i],delta_pp=d[i]))
    for name,rows in [("arm_summary.csv",arms),("endpoint_summary.csv",endpoints),("endpoint_pairs.csv",pairs),("crossing_summary.csv",crosses),("crossing_by_teacher.csv",byseed)]:
        write(name,rows)

if __name__=="__main__":
    main()
