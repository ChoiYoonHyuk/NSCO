"""Summarize the frozen cyclic-shift control. Only per-teacher integer counts are needed."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
spec = json.loads((root/"control_plan.json").read_text())
out = root/"results"
tests = spec["test"]
rx = {}
for seed in spec["seeds"]:
    for row in csv.DictReader((out/f"teacher_{seed}.csv").open(newline="")):
        k = (int(row["seed"]), float(row["rate"]), row["arm"], int(row["examples"]))
        assert k[0] == seed and int(row["depth"]) == spec["depth"] and int(row["test"]) == tests
        assert int(row["labels"]) == k[3] * (1 + int(k[2]!="final")) and k not in rx
        assert 0 <= int(row["errors"]) <= tests and 0 <= int(row["path_hits"]) <= tests
        rx[k] = int(row["errors"])
expected=set(itertools.product(spec["seeds"],spec["rates"],spec["arms"],spec["checkpoints"]))
assert set(rx) == expected
rng = np.random.default_rng(726099)
idx = rng.integers(len(spec["seeds"]),size=(50000,len(spec["seeds"])))
comparisons = []
pairs = []
for rate in spec["rates"]:
    byarm = {arm:np.array([rx[seed,rate,arm,spec["records"]] for seed in spec["seeds"]])*100/tests
             for arm in spec["arms"]}
    for i, seed in enumerate(spec["seeds"]):
        pairs.append(dict(seed=seed, rate=rate, final=byarm["final"][i], true=byarm["true"][i],
                          shift=byarm["shift"][i], true_minus_shift_pp=byarm["true"][i]-byarm["shift"][i]))
    for arm, ref in [("true","shift"),("true","final"),("shift","final")]:
        delta = byarm[arm]-byarm[ref]
        lo, hi = np.quantile(delta[idx].mean(1), [.025,.975])
        wins, losses = int((delta<0).sum()), int((delta>0).sum())
        n = wins + losses
        p = min(1.,2*sum(math.comb(n,j) for j in range(min(wins,losses)+1))/2**n)
        comparisons.append(dict(rate=rate, arm=arm, ref=ref, arm_error_pct=byarm[arm].mean(),
             ref_error_pct=byarm[ref].mean(), delta_pp=delta.mean(), ci_low_pp=lo, ci_high_pp=hi,
             n_lower=wins, n_higher=losses, n_tied=len(delta)-n, sign_p=p))
for name,rows in [("comparisons.csv",comparisons),("teacher_endpoints.csv",pairs)]:
    with (out/name).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
print((out/"comparisons.csv").read_text())
meta=json.loads((out/"provenance.json").read_text())
for f in ["control_plan.json","control.py","deep_adam.py","gaussian.py","plan.json"]:
    assert hashlib.sha256((root/f).read_bytes()).hexdigest()==meta[f]
