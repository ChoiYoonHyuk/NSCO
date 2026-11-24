"""Audit and summarize common-target test measurements with teacher as the paired unit."""
from pathlib import Path
import csv
import json
from math import comb
import hashlib
import numpy as np

root=Path(__file__).resolve().parent
plan=json.loads((root/"readout_plan.json").read_text("utf8"))
keys={}
mets=["path_hits","parity_errors","maj3_errors"]
for f in sorted((root/"results").glob("teacher_*.csv")):
    for row in csv.DictReader(f.open(encoding="utf8")):
        key=(int(row["seed"]),float(row["rate"]),int(row["n"]),row["arm"])
        assert key not in keys and int(row["test"])==8192
        vals=[int(row[k]) for k in mets]
        assert all(0<=v<=8192 for v in vals)
        assert vals[0]+max(vals[1:])<=8192
        keys[key]=vals
assert set(keys)=={(s,r,n,a) for s in plan["teachers"] for r in plan["rates"] for n in plan["checkpoints"] for a in plan["arms"]}
meta=json.loads((root/"results/provenance.json").read_text())
for src in ("readout.py","readout_plan.json","gaussian.py","deep_adam.py","plan.json"):
    assert hashlib.sha256((root/src).read_bytes()).hexdigest()==meta[src]
ix=np.random.default_rng(7534017).integers(0,16,(50000,16))
rows=[]
for r in plan["rates"]:
    for n in plan["checkpoints"]:
        p,m=[np.array([keys[s,r,n,a] for s in plan["teachers"]])*100/8192 for a in plan["arms"]]
        for i,metric in enumerate(mets):
            delta=m[:,i]-p[:,i]
            lo,hi=np.percentile(delta[ix].mean(axis=1),[2.5,97.5])
            pos,neg=int(sum(delta>0)),int(sum(delta<0))
            signp=min(1,2*sum(comb(pos+neg,k) for k in range(min(pos,neg)+1))/2**(pos+neg))
            rows.append(dict(rate=r,n=n,metric=metric,parity_training=p[:,i].mean(),maj3_training=m[:,i].mean(),delta=delta.mean(),lo=lo,hi=hi,positive=pos,negative=neg,sign_p=signp))
            if n==plan["examples"]:
                print(r,metric,*(round(v,5) for v in [p[:,i].mean(),m[:,i].mean(),delta.mean(),lo,hi]),pos,neg,signp,flush=True)
with (root/"results/summary.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
