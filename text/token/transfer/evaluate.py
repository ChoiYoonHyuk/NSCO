"""Evaluate frozen three-hop models at new lengths/depths. No optimizer or training."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

HERE = Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def reference_links(x):
    n, length=x.shape
    seen=np.full((n,4),-2,int)
    pointer=np.empty_like(x)
    b=np.arange(n)
    for t in range(length):
        col=x[:,t]
        pointer[:,t]=seen[b,col]+1
        seen[b,col]=t
    return pointer

def path_at_all_positions(p,hops):
    b=np.arange(len(p))[:,None]
    steps=[p]
    for _ in range(1,hops):
        old=steps[-1]
        steps.append(np.where(old<0,-1,p[b,np.maximum(old,0)]))
    return np.stack(steps,-1)

def rollout(x,q,hops,mode,match=None,decay=None):
    b=np.arange(len(x))
    cols=np.arange(1,x.shape[1])
    t=q.copy()
    trajectory=[]
    for _ in range(hops):
        if mode=="rule":
            # Direct reverse search via the latest matching original token.
            allpos=np.arange(x.shape[1])
            eq=x==x[b,t,None]
            last=np.where(eq & (allpos<t[:,None]),allpos,-2).max(1)
            t=last+1
            assert np.min(t)>=0
        elif mode=="recency":
            t=np.maximum(0,t-1)
        else:
            key=x[:,cols-int(mode=="successor")]
            scores=match[x[b,t,None],key]-decay*np.log1p(np.maximum(0,t[:,None]-cols))
            scores[cols>=t[:,None]]=-np.inf
            t=np.where(t<2,0,1+scores.argmax(1))
        trajectory.append(t.copy())
    return np.stack(trajectory,1)

def score(s,truth,pred):
    b=np.arange(len(s))[:,None]
    exact=np.all(pred==truth,axis=1)
    wrong=s[b,pred].prod(1)!=s[b,truth].prod(1)
    # Ground truth has distinct positions. Predicted positions are decreasing
    # except possible repeated zero fallbacks. Equal Boolean parity monomials
    # then require identical ordered paths. Check independently with XOR masks.
    equal=[]
    for t,p in zip(truth,pred):
        key_t=key_p=0
        for j in t:
            key_t ^= 1<<int(j)
        for j in p:
            key_p ^= 1<<int(j)
        equal.append(key_t==key_p)
    np.testing.assert_array_equal(exact,equal)
    return dict(correct_paths=int(exact.sum()),sign_mistakes=int(wrong.sum()),
                twice_expected_sign_mistakes=int((~exact).sum()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("shifted",type=Path,help="original frozen fit directory")
    args=ap.parse_args()
    p=json.loads((HERE/"protocol.json").read_text())
    prior=json.loads((args.shifted/"protocol.json").read_text())
    fits=json.loads((args.shifted/"audit.json").read_text())["frames"]
    arms=[(a,g) for a in prior["alignment"] for g in prior["readouts"]]
    rows=[]
    for seed,stream in zip(p["fit_seeds"],p["new_input_seeds"]):
        f=args.shifted/f"fits_{seed}.npz"
        assert sha(f)==next(z["frames_sha256"] for z in fits if z["seed"]==seed)
        net={}
        with np.load(f,allow_pickle=False) as saved:
            for kind in ["successor","same_position"]:
                a=arms.index((kind,"parity"))
                net[kind]=(saved["query"][a]@saved["key"][a].T/2.,
                           np.logaddexp(0,saved["decay"][a]))
        rng=np.random.RandomState(stream)
        n=p["sequences_per_seed"]
        increments=rng.randint(1,4,(n,max(p["lengths"])))
        increments[:,0]=rng.randint(0,4,n)
        tokens=np.cumsum(increments,axis=1)%4
        signs=2*rng.randint(2,size=tokens.shape)-1
        for length in p["lengths"]:
            x=tokens[:,:length]
            allpaths=path_at_all_positions(reference_links(x),max(p["depths"]))
            eligible=(allpaths[:,:,-1]>=0)&(np.arange(length)>=length//2)
            assert np.all(eligible.any(1))
            q=np.where(eligible,rng.random_sample(x.shape),-1).argmax(1)
            truth=allpaths[np.arange(n),q]
            for mode in ("rule","recency","successor","same_position"):
                fit=net.get(mode,(None,None))
                pred=rollout(x,q,max(p["depths"]),mode,*fit)
                for k in p["depths"]:
                    counts=score(signs[:,:length],truth[:,:k],pred[:,:k])
                    if mode=="rule":
                        assert counts["correct_paths"]==n and counts["sign_mistakes"]==0
                    rows.append(dict(seed=seed,stream=stream,length=length,depth=k,model=mode,examples=n,**counts))
    with (HERE/"counts.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary=[]
    for length in p["lengths"]:
        for k in p["depths"]:
            for mode in ("rule","recency","successor","same_position"):
                group=[x for x in rows if (x["length"],x["depth"],x["model"])==(length,k,mode)]
                assert len(group)==5
                sums={key:sum(x[key] for x in group) for key in ("examples","correct_paths","sign_mistakes","twice_expected_sign_mistakes")}
                rates=[x["correct_paths"]/x["examples"] for x in group]
                margin=2.776445105*statistics.stdev(rates)/math.sqrt(len(rates))
                summary.append(dict(length=length,depth=k,model=mode,**sums,per_seed_path_rates=rates,
                                    seed_t95=[statistics.mean(rates)-margin,statistics.mean(rates)+margin]))
    (HERE/"summary.json").write_text(json.dumps(dict(rows=summary,epochs=0,protocol_sha256=sha(HERE/"protocol.json"),
        source_sha256=sha(Path(__file__)),counts_sha256=sha(HERE/"counts.csv")),indent=2)+"\n")
    for row in summary:
        print(row["length"],row["depth"],row["model"],row["correct_paths"],row["sign_mistakes"])

if __name__=="__main__":
    main()
