"""One-pass paired full-support readout experiment; plans and metrics are local to this directory."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import time

import torch
import deep_adam as old
import gaussian as base

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT/"readout_plan.json").read_text(encoding="utf-8-sig"))
OUT = ROOT/"results"
OUT.mkdir(exist_ok=True)


def readouts(signs):
    x,y,z = signs.reshape(*signs.shape[:-1],-1,3).unbind(-1)
    return signs.prod(-1), ((x+y+z-x*y*z)/2).prod(-1)


def path_and_signs(frames, query, signs, keys, soft):
    batch,depth=signs.shape[:2]
    projected=(keys.reshape(-1,15) @ frames).reshape(len(frames),batch,depth,2,2)
    state=query[None]
    values, choices=[], []
    for hop in range(depth):
        first,second=projected[:,:,hop].unbind(2)
        diff=second-first
        score=(state*diff).sum(-1)
        weight=torch.sigmoid(base.THETA*score) if soft else (score>0).to(query.dtype)
        values.append(signs[:,hop,0]+weight*(signs[:,hop,1]-signs[:,hop,0]))
        choices.append(weight)
        state=first+weight[...,None]*diff
    return torch.stack(choices,-1), torch.stack(values,-1)


def teacher_outputs(data):
    _,s,_,par,_,path,y=data
    selected=s.gather(-1,path.long().unsqueeze(-1)).squeeze(-1)
    hp,hm=readouts(selected)
    return hp, hm, y*par


def loss_gradient(frames,is_major,data):
    q,s,key,*_=data
    _, softs=path_and_signs(frames,q,s,key,True)
    p,m=readouts(softs)
    hp,hm,flip=teacher_outputs(data)
    residual=torch.where(is_major[:,None],m-hm*flip,p-hp*flip)
    return torch.autograd.grad(.5*residual.square().mean(1).sum(),frames)[0]


@torch.no_grad()
def evaluate(frames,data):
    q,s,key,par,_,true_path,_=data
    tp,tm,_=teacher_outputs(data)
    counts=torch.zeros(len(frames),3,device=base.DEVICE,dtype=torch.long)
    frames=base.polar(frames)
    for start in range(0,len(q),1024):
        sl=slice(start,start+1024)
        path,signs=path_and_signs(frames,q[sl],s[sl],key[sl],False)
        predp,predm=readouts(signs)
        counts[:,0]+=(path==true_path[sl]).all(-1).sum(1)
        counts[:,1]+=(predp!=tp[sl]).sum(1)
        counts[:,2]+=(predm!=tm[sl]).sum(1)
    return counts.tolist()


def checks():
    seed=753300
    U,_=base.configs(seed,[(2,0,1,1)]*2)
    teacher=base.frame(seed)
    data=old.draw(256,PLAN["depth"],base.generator(seed,3),teacher)
    q,s,k,par,c1,path,observed=data
    tp,tm,flip=teacher_outputs(data)
    assert torch.equal(tp,par)
    assert torch.equal(observed,tp*flip)
    triplets=torch.tensor(list(itertools.product((-1.,1.),repeat=3)),device=base.DEVICE)
    assert torch.equal(readouts(triplets)[1],triplets.sum(-1).sign())
    predicted_path,values=path_and_signs(teacher[None],q,s,k,False)
    assert torch.equal(path,predicted_path[0])
    assert torch.equal(c1,values[0,:,0])
    assert evaluate(teacher[None],data)==[[256,0,0]]
    cond=torch.zeros(2,dtype=torch.bool,device=base.DEVICE)
    a=loss_gradient(U,cond,data)
    b=old.gradient(U,cond,data)
    torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-7)
    return (a-b).abs().max().item()


def run(seed):
    cfg=list(itertools.product(PLAN["rates"],PLAN["arms"]))
    U,_=base.configs(seed,[(2,0,1,1)]*len(cfg))
    eta=torch.tensor([r for r,a in cfg],device=base.DEVICE)[:,None,None]
    which=torch.tensor([a=="maj3" for r,a in cfg],device=base.DEVICE)
    teacher=base.frame(seed)
    train=base.generator(seed,3)
    testdata=old.draw(PLAN["test"],PLAN["depth"],base.generator(seed,5),teacher)
    m,v=torch.zeros_like(U),torch.zeros_like(U)
    rows=[]
    for step in range(1,PLAN["examples"]//PLAN["batch"]+1):
        obs=old.draw(PLAN["batch"],PLAN["depth"],train,teacher)
        grad=loss_gradient(U,which,obs)
        with torch.no_grad():
            grad/=grad.norm(dim=(-2,-1),keepdim=True).clamp_min(1.)
            m.mul_(.9).add_(grad,alpha=.1)
            v.mul_(.999).addcmul_(grad,grad,value=.001)
            direction=(m/(1-.9**step))/((v/(1-.999**step)).sqrt()+1e-8)
            U-=(eta/math.sqrt(1+(step-1)/128.))*direction
        n=step*PLAN["batch"]
        if n in PLAN["checkpoints"]:
            for (rate,arm),(hits,ep,em) in zip(cfg,evaluate(U,testdata)):
                rows.append(dict(seed=seed,rate=rate,arm=arm,n=n,test=PLAN["test"],path_hits=hits,parity_errors=ep,maj3_errors=em))
    return rows


if __name__=="__main__":
    maxdiff=checks()
    meta={"model":torch.cuda.get_device_name(),"torch":str(torch.__version__),"cuda":torch.version.cuda,"baseline_parity_gradient_maxerr":maxdiff,"start_unix":time.time()}
    for name in ["readout.py","readout_plan.json","deep_adam.py","gaussian.py","plan.json"]:
        meta[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (OUT/"provenance.json").write_text(json.dumps(meta,indent=2))
    for seed in PLAN["teachers"]:
        file=OUT/f"teacher_{seed}.csv"
        if file.exists(): continue
        rows=run(seed)
        tmp=file.with_suffix(".part")
        with tmp.open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
        tmp.replace(file)
        print("Complete teacher",seed,flush=True)

