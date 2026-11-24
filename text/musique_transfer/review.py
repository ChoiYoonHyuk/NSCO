"""Zero-update evaluation, exhaustive clean risks, paired source-stratified intervals."""
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import numpy as np
from holdout import HERE, BRIDGE, legacy, read, save
from audit import direct_replay, signs_error

TASKS=["first3","last3","parity3","majority"]
ARMS=["frozen","e5","final","first_value","first_index","all_indices"]
METRICS=["error","path_em","support_f1"]

def scores(f,e,n,g,w,task):
    chosen=direct_replay(f,e,n,w,3)
    assert np.all(np.diff(np.sort(chosen,axis=1),axis=1)>0) and np.all(chosen<n[:,None])
    overlap=(chosen[:,:,None]==g[:,None,:]).any(-1).sum(-1)
    q=task.rstrip("3")
    exact={"first":.5*(chosen[:,0]!=g[:,0]),"last":.5*(chosen[:,-1]!=g[:,-1]),"parity":.5*(overlap!=3),
           "majority":.5*(1-overlap/4-(overlap==3)/4)}[q]
    for i,(gt,pred) in enumerate(zip(g.tolist(),chosen.tolist())):
        m={x:j for j,x in enumerate(dict.fromkeys(gt+pred))}
        assert signs_error(q,3,tuple(m[x] for x in pred),tuple(m[x] for x in gt))==exact[i]
    return np.column_stack((exact,(chosen==g).all(-1),overlap/3))

def contrast_draws(src, first_gold, clustered=False):
    rng=np.random.default_rng(5845993)
    weights=np.zeros((5000,len(src)),np.float32)
    for source in np.unique(src):
        ids=np.flatnonzero(src==source)
        units=(first_gold[ids] if clustered else np.arange(len(ids)))
        unique,grp=np.unique(units,return_inverse=True)
        sampled=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=5000)
        # Each original question retains weight one within a resampled cluster.
        local=sampled[:,grp]
        weights[:,ids]=local/local.sum(-1,keepdims=True)*(len(ids)/len(src))
    return weights

def main():
    exp=read(HERE/"exposure.json"); groups=read(HERE/"private.groups.json")
    assert exp["registration_sha256"]==legacy.sha(HERE/"registration.json")
    manifest=read(HERE/"private.manifest.json")
    data, sources, fsupport = [],[],[]
    cohorts=defaultdict(list); raw_read={}
    for src in ["dev","train"]:
        rows=legacy.load_rows(src)
        raw_read[src]=rows
        ids=manifest[src]["indices"]; base=len(sources)
        if not ids:continue
        with np.load(HERE/f"{src}_3.private.npz") as a:
            data.append({k:a[k] for k in a})
        for idx in ids:
            r=rows[idx]; cand=sorted(r["paragraphs"],key=lambda p:p["idx"]); t=cand[r["question_decomposition"][0]["paragraph_support_idx"]]
            fsupport.append(legacy.text_key(t["paragraph_text"])); sources.append(src)
        for tier,subset in groups[src].items():
            chosen=[base+i for i in subset]
            cohorts[src+"_"+tier].extend(chosen)
            cohorts["pooled_"+tier].extend(chosen)
    src=np.array(sources); first=np.array(fsupport)
    f,e,n,g=(np.concatenate([x[k] for x in data]) for k in ("f","e","n","g"))
    cfg=read(BRIDGE/"protocol.json"); seeds=cfg["seeds"]
    outcomes={}
    for draw,folder in [("eight","."),("one","one_draw")]:
        md=BRIDGE/folder
        lock=read(md/"selection_lock.json")
        assert exp["model_lock_sha256"][draw]==legacy.sha(md/"selection_lock.json")==read(md/"evaluated.json")["selection_sha256"]
        for task in TASKS:
            cell=lock["choices"][task]
            base=np.array(cfg["frozen_configs"][cell["frozen"]],np.float32)
            for scale in ["selected","raw"]:
                for arm in ARMS:
                    vals=[]
                    for seed in seeds:
                        if arm=="e5":w=np.array(cfg["frozen_configs"]["e5"],np.float32)
                        elif arm=="frozen":w=base
                        else:
                            st=cell["seeds"][str(seed)]["arms"][arm]
                            raw=np.array(st["raw_weights"],np.float32)
                            w=base+st["scale"]*(raw-base) if scale=="selected" else raw
                        vals.append(scores(f,e,n,g,w,task))
                    outcomes[task,draw,scale,arm]=np.stack(vals)
    summary={}
    for name,ids in cohorts.items():
        if not ids:
            summary[name]={"n":0,"status":"empty"};continue
        ysrc=src[ids]; yg=first[ids]
        W=contrast_draws(ysrc,yg); CW=contrast_draws(ysrc,yg,True)
        D={}; measure={}
        for task in TASKS:
            for draw in ["one","eight"]:
                for scale in ["selected","raw"]:
                    for arm in ARMS:
                        val=outcomes[task,draw,scale,arm][:,ids,:]
                        stats=val.mean(1)
                        measure["/".join([task,draw,scale,arm])]={m:{"mean":float(stats[:,i].mean()),"seed_sd":float(stats[:,i].std(ddof=1))} for i,m in enumerate(METRICS)}
                    for arm,reference in [("first_index","first_value"),("first_index","final"),("first_value","final"),("final","frozen"),("all_indices","frozen"),("first_index","frozen"),("first_value","frozen")]:
                        delta=(outcomes[task,draw,scale,arm]-outcomes[task,draw,scale,reference])[:,ids,:]
                        D["/".join([task,draw,scale,arm+"-"+reference])]=delta
            for scale in ["selected","raw"]:
                a=D["/".join([task,"eight",scale,"first_index-first_value"])]
                b=D["/".join([task,"one",scale,"first_index-first_value"])]
                D["/".join([task,"interaction",scale,"first_index-first_value"])]=a-b
        keys=list(D)
        # Only the seed-averaged question differences enter either bootstrap.
        matrix=np.column_stack([D[key].mean(0) for key in keys]).astype(np.float32)
        ci=np.quantile(W@matrix,[.025,.975],axis=0).reshape(2,len(keys),3)
        cci=np.quantile(CW@matrix,[.025,.975],axis=0).reshape(2,len(keys),3)
        diffs={}
        for j,key in enumerate(keys):
            a=D[key].mean(1)
            diffs[key]={m:{"mean":float(a[:,i].mean()),"seed_values":a[:,i].tolist(),"q_ci95":ci[:,j,i].tolist(),"first_support_cluster_ci95":cci[:,j,i].tolist()} for i,m in enumerate(METRICS)}
        summary[name]={"n":len(ids),"unique_first_supports":{s:len(set(yg[ysrc==s])) for s in set(ysrc)},"models":measure,"differences":diffs}
        print(name,len(ids),diffs["majority/interaction/selected/first_index-first_value"]["error"],flush=True)
    save("aggregate.json",{"registration_sha256":exp["registration_sha256"],"cohorts":summary,"training_epochs":0,
                        "new_fit_or_selection_labels":0,"all_risks_match_exhaustive_enumeration":True,"metrics":METRICS})

if __name__=="__main__":main()
