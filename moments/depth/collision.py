"""Independent check of cross-frame correlations using exact conditional sign integration."""
import json
import numpy as np
from pathlib import Path
root=Path(__file__).resolve().parent
ks=(3,6,9);seeds=(495011,495017,495037,495049,495053,495071,495089,495107)
per_seed=2**16
stats={(k,f):[] for k in ks for f in ("parity","majority","terminal")}
observed={key:[] for key in stats}
for s in seeds:
    r=np.random.default_rng(s)
    sr=np.random.default_rng(np.random.SeedSequence([s,191]))
    totals={key:0. for key in stats}
    direct={key:0. for key in stats}
    for off in range(0,per_seed,4096):
        n=min(4096,per_seed-off)
        q=r.normal(size=(n,2));q/=np.linalg.norm(q,axis=1,keepdims=True)
        # Orthogonal target frames pick disjoint pairs of coordinates from
        # each actual Gaussian record; the two paths start at the same query.
        state=[q,q]; hits=[];selected=[[],[]];row=np.arange(n)
        for h in range(9):
            z=r.normal(size=(n,2,4));i=[]
            for j in (0,1):
                key=z[:,:,2*j:2*j+2]
                delta=key[:,1]-key[:,0]
                choice=(np.einsum('bi,bi->b',delta,state[j])>0).astype(int)
                state[j]=key[row,choice];i.append(choice)
            hits.append((i[0]==i[1]).astype(float))
            signs=2*sr.integers(2,size=(n,2),dtype=np.int8)-1
            for j in (0,1):selected[j].append(signs[row,i[j]])
            k=h+1
            if k not in ks:continue
            mat=np.array(hits).T
            groups=mat.reshape(n,k//3,3)
            conditional_majority=((groups.sum(axis=-1)+groups.prod(axis=-1))/4).prod(axis=1)
            totals[k,"majority"]+=conditional_majority.sum()
            totals[k,"parity"]+=mat.prod(axis=1).sum()
            totals[k,"terminal"]+=mat[:,-1].sum()
            v0,v1=(np.array(sels).T for sels in selected)
            direct[k,"parity"]+=(v0.prod(1)*v1.prod(1)).sum()
            direct[k,"majority"]+=(np.sign(v0.reshape(n,k//3,3).sum(2)).prod(1)*np.sign(v1.reshape(n,k//3,3).sum(2)).prod(1)).sum()
            direct[k,"terminal"]+=(v0[:,-1]*v1[:,-1]).sum()
    for key,v in totals.items():
        stats[key].append(v/per_seed)
        observed[key].append(direct[key]/per_seed)
result={"per_seed":per_seed,"seeds":seeds,"total_paths":len(seeds)*per_seed,"rows":[]}
for (k,f),v in stats.items():
    expected=2**(-k) if f=="parity" else (13/32)**(k//3) if f=="majority" else .5
    a=np.asarray(v)
    result["rows"].append({"k":k,"family":f,"estimate":float(a.mean()),"expected":expected,
      "seed_se":float(a.std(ddof=1)/np.sqrt(len(seeds))),"seed_estimates":v,
      "direct_readout_correlation":float(np.mean(observed[k,f])),"direct_seed_estimates":observed[k,f]})
(root/"collision.json").write_text(json.dumps(result,indent=2))
