"""Unordered set scoring, analytic shared-sign error and cluster-aware inference."""
from collections import Counter,defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
import numpy as np
from candidates import ROOT,P,put,sha


def greedy(f,e,w):
    unary=np.einsum("bnf,f->bn",f,w[:4],optimize=False)
    edge=np.einsum("bijf,f->bij",e,w[4:],optimize=False)
    n=len(f);row=np.arange(n)
    seen=np.zeros((n,20),bool);selected=[]
    for h in range(3):
        ix=np.where(seen,-np.inf,unary).argmax(-1)
        selected.append(ix);seen[row,ix]=True
        unary=edge[row,ix]
    return np.stack(selected,axis=1)


def cpu_replay(f,e,w):
    result=[]
    for direct,trans in zip(f,e):
        # Avoid a BLAS rounding change when scores are very close.
        scores=(direct*w[:4]).sum(-1);path=[]
        for _ in range(3):
            ix=max((j for j in range(20) if j not in path),key=lambda j:scores[j])
            path.append(ix)
            scores=(trans[ix]*w[4:]).sum(-1)
        result.append(path)
    return np.array(result)


def majority_check():
    ans={}
    for r in range(4):
        g=[0,1,2];sel=g[:r]+list(range(3,6-r))
        bits=np.array(list(itertools.product([-1,1],repeat=6-r)))
        err=float((np.sign(bits[:,g].sum(-1))!=np.sign(bits[:,sel].sum(-1))).mean())
        assert err==.5*(1-r/4-int(r==3)/4)
        ans[r]=err
    return ans


def write(name, rows):
    with (ROOT/name).open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)


def main():
    c=json.loads((ROOT/"candidate_receipt.json").read_text());feat=json.loads((ROOT/"feature_receipt.json").read_text())
    assert c["protocol_sha256"]==sha(ROOT/"protocol.json") and c["frozen_weights_sha256"]==sha(ROOT/"weights.json")
    assert c["labels_hash"]==sha(ROOT/"labels.private.npz") and feat["feature_sha256"]==sha(ROOT/"features.private.npz")
    x=np.load(ROOT/"features.private.npz");f,e=x["f"],x["e"]
    labels=np.load(ROOT/"labels.private.npz");g=labels["g"];group=labels["cluster"];bm=labels["bm25"]
    assert np.all(g[:,1:]>g[:,:-1]) and (g>=0).all() and (g<20).all()
    # This comparison observes only an unordered set. The gold order is deliberately arbitrary.
    def overlap(pred):
        return (pred[:,:,None]==g[:,None,:]).any(-1).sum(-1)
    def metrics(r):
        return {"doc_f1":r/3.,"exact_set":(r==3).astype(float),
                "majority_error":.5*(1-r/4.-(r==3)/4.)}
    weights=json.loads((ROOT/"weights.json").read_text());cells={};rows=[];mismatches=0;permdiff=0
    perm=np.random.default_rng(50123).permutation(20)
    perms_checked=0
    for budget,conf in weights.items():
        w0=np.array(conf["base"],np.float32)
        methods=["bm25","e5_direct","bge_direct","frozen","final","first_value","first_index","all_indices"]
        for arm in methods:
            cell=[]
            for seed in P["seeds"]:
                if arm in ["bm25","e5_direct","bge_direct"]:
                    scores={"bm25":bm,"e5_direct":f[:,:,0],"bge_direct":f[:,:,1]}[arm]
                    pred=np.argsort(-scores,kind="stable")[:,:3]
                else:
                    if arm=="frozen":w=w0
                    else:
                        prm=conf["seeds"][str(seed)][arm]
                        raw=np.array(prm["raw"],np.float32)
                        w=w0+np.float32(prm["scale"])*(raw-w0)
                    pred=greedy(f,e,w)
                    alt=cpu_replay(f,e,w)
                    mismatches+=int((alt!=pred).any(1).sum())
                    shuffled=greedy(f[:,perm],e[:,perm][:,:,perm],w)
                    perms_checked+=len(pred)
                    permdiff+=int((perm[shuffled]!=pred).any(1).sum())
                assert np.all(np.diff(np.sort(pred,axis=1),axis=1)>0)
                r=overlap(pred);cell.append(r)
                met=metrics(r);rows.append(dict(draws=budget,arm=arm,seed=seed,
                    **{k:float(v.mean()) for k,v in met.items()},
                    **{f"overlap_{j}":int((r==j).sum()) for j in range(4)}))
            cells[budget,arm]=np.array(cell)
    write("seed_metrics.csv",rows)
    # Save binned, anonymous cluster sufficient statistics to reproduce the primary CI.
    # Using a canonical sorted list makes the released histogram sufficient for exact replay.
    _,inv=np.unique(group,return_inverse=True)
    sizes=np.bincount(inv)
    def estimate(delta,primary=False):
        sums=np.bincount(inv,weights=delta)
        pairs=sorted(zip(sizes,sums))
        pairs=np.array(pairs)
        draw=np.random.default_rng(P["bootstrap_seed"]).integers(0,len(pairs),(P["bootstrap"],len(pairs)))
        rep=pairs[draw,1].sum(1)/pairs[draw,0].sum(1)
        if primary:
            # For F1 = r/3 and 5 paired seeds, integer numerator equals delta*15.
            hist=Counter((int(a),round(float(b)*15)) for a,b in pairs)
            put("primary_cluster_histogram.json",[dict(claims=a,overlap_diff_sum=b,clusters=n)
                for (a,b),n in sorted(hist.items())])
        return float(delta.mean()),*np.quantile(rep,[.025,.975]).tolist()
    contrasts=[]
    for (budget,arm),rs in cells.items():
        for ref in (["frozen","first_value"] if arm!="first_value" else ["frozen"]):
            if ref==arm:continue
            V=metrics(rs);base=metrics(cells[budget,ref])
            for metric in V:
                delta=(V[metric]-base[metric]).mean(0)
                primary=(budget,arm,ref,metric)==("eight","first_index","first_value","doc_f1")
                mean,lo,hi=estimate(delta,primary)
                contrasts.append(dict(draws=budget,arm=arm,against=ref,metric=metric,
                                     difference=mean,cluster95_low=lo,cluster95_high=hi,primary=primary))
    write("contrasts.csv",contrasts)
    means=[]
    for (budget,arm),rs in cells.items():
        vals=metrics(rs)
        means.append(dict(draws=budget,arm=arm,**{k:float(v.mean()) for k,v in vals.items()}))
    write("aggregate.csv",means)
    hyper=np.array([math.comb(3,i)*math.comb(17,3-i)/math.comb(20,3) for i in range(4)])
    exact=majority_check()
    put("score_receipt.json",dict(training_epochs=0,validation_examples=0,exact_formula_check=exact,
        uniform_random=dict(doc_f1=.15,exact_set=1/math.comb(20,3),
                            majority_error=sum(hyper[j]*exact[j] for j in range(4))),
        independent_rollout_mismatches=mismatches,permutation_changes=permdiff,
        permutation_predictions_checked=perms_checked,code_sha256=sha(Path(__file__)),
        candidate_receipt=sha(ROOT/"candidate_receipt.json"),feature_receipt=sha(ROOT/"feature_receipt.json"),
        numeric_sha256={k:sha(ROOT/k) for k in ["seed_metrics.csv","aggregate.csv","contrasts.csv","primary_cluster_histogram.json"]}))
    assert mismatches==0 and permdiff==0,(mismatches,permdiff)
    print([v for v in contrasts if v["primary"]],flush=True)


if __name__=="__main__":main()
