"""Within-seed supervision contrasts and un-interpolated grid sensitivity."""
from pathlib import Path
import csv
import json
import numpy as np
root=Path(__file__).resolve().parent
records=list(csv.DictReader((root/"crossings.csv").open()))
keys={(int(row["seed"]),int(row["k"]),row["family"],row["supervision"]):row for row in records}
assert all(row["censoring"]=="bracket" for row in records)
seeds=sorted({a[0] for a in keys});ds=np.array([3,6,9]);dc=ds-ds.mean()
ids=np.random.default_rng(412003).integers(len(seeds),size=(10000,len(seeds)))
def summary(v,ratio=False):
    f=np.exp2 if ratio else np.asarray
    lo,hi=np.quantile(f(v[ids].mean(axis=1)),[.025,.975])
    return [float(f(v.mean())),float(lo),float(hi)]
out={}
for name in ("bracket","grid"):
    arr={}
    def value(row):
        return float(row["log2_interpolated_crossing"]) if name=="bracket" else np.log2(int(row["last_tested_total_labels"]))
    for family,mode in sorted({(k[2],k[3]) for k in keys}):
        arr[family,mode]=np.array([[value(keys[s,k,family,mode]) for k in ds] for s in seeds])
    result={"slopes":{},"excess_slopes":{},"k9_cost_ratios":{},"geometric_mean_labels":{}}
    for (family,mode),x in arr.items():
        result["slopes"][family+"/"+mode]=summary(x@dc/(dc@dc))
        result["geometric_mean_labels"][family+"/"+mode]=[float(z) for z in np.exp2(x.mean(0))]
    for family in ("parity","majority_product"):
        delta=arr[family,"final"]-arr[family,"plus_C1"]
        result["excess_slopes"][family]=summary(delta@dc/(dc@dc))
        result["k9_cost_ratios"][family]=summary(delta[:,-1],True)
    out[name]=result
(root/"paired_summary.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out))
