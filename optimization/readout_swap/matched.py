"""One-pass Gaussian SGD/Adam with matched readout degree profiles."""
import csv
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
P = json.loads((ROOT/"protocol.json").read_text())
NAMES = ["A","B","A","B"]
METHODS = ["SGD","SGD","Adam","Adam"]
DEV = "cuda"
torch.set_num_threads(2)
torch.set_float32_matmul_precision("highest")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, rows):
    with (OUT/name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def rng(seed, stream):
    return torch.Generator(device=DEV).manual_seed(seed*100+P["streams"][stream])


def maj(a,b,c):
    return (a+b+c-a*b*c)/2


def read(v):
    a,b,c,d = v.unbind(-1)
    return torch.stack([a*maj(b,c,d),b*maj(a,c,d)],-1)


def network(frames,q,s,z,soft):
    batch=len(q)
    m=len(frames)
    keys=(z.reshape(-1,31)@frames).reshape(m,batch,4,2,4)
    query=q[None]
    values, selection=[],[]
    for h in range(4):
        left,right=keys[:,:,h].unbind(2)
        diff=right-left
        gap=(diff*query).sum(-1)
        w=(4*gap).sigmoid() if soft else (gap>0).to(q.dtype)
        values.append(s[:,h,0]+w*(s[:,h,1]-s[:,h,0]))
        selection.append(w)
        query=left+w[...,None]*diff
    return read(torch.stack(values,-1)),torch.stack(selection,-1)


def draw(gen,V,n):
    q=torch.nn.functional.normalize(torch.randn(n,4,device=DEV,generator=gen))
    s=(torch.randn(n,4,2,device=DEV,generator=gen)>=0).float()*2-1
    z=torch.randn(n,4,2,31,device=DEV,generator=gen)
    clean, paths=network(V[None],q,s,z,False)
    flip=(torch.rand(n,device=DEV,generator=gen)>=P["flip_probability"]).float()*2-1
    return q,s,z,clean[0]*flip[:,None],clean[0],paths[0]


def gradient(U,data):
    q,s,z,Y,_,_=data
    preds,_=network(U,q,s,z,True)
    response=torch.stack([preds[i,:,i%2] for i in range(4)])
    targets=Y.T.repeat(2,1)
    return torch.autograd.grad(0.5*(response-targets).square().mean(1).sum(),U)[0]


def polar(x):
    a,_,b=torch.linalg.svd(x,full_matrices=False)
    return a@b


def probe(seed,V,U):
    gen=rng(seed,"gradient")
    samples=torch.stack([gradient(U,draw(gen,V,P["batch"]))[:2] for _ in range(P["probe_batches"])]).double()
    m=samples.mean(0)
    var=samples.var(0,unbiased=True).sum((-1,-2))
    sq=m.square().sum((-1,-2))-var/len(samples)
    return [dict(seed=seed,readout=t,mean_norm=float(m[i].norm()),noise_rms=float(var[i].sqrt()),
                 unbiased_signal_sq=float(sq[i]),snr_squared=float(sq[i]/var[i])) for i,t in enumerate(["A","B"])]


def score(seed,n,U,V,test):
    q,s,z,_,clean,hops=test
    frames=torch.cat([U,polar(U)])
    counts=torch.zeros(8,6,device=DEV)
    for a in range(0,len(q),256):
        p=slice(a,a+256)
        pred,path=network(frames,q[p],s[p],z[p],False)
        error=torch.stack([pred[j,:,j%2]!=clean[p,j%2] for j in range(8)])
        hit=path==hops[p]
        counts+=torch.cat([error.sum(1)[:,None],hit.sum(1),hit.all(-1).sum(1)[:,None]],1)
    proj=polar(frames)
    dif=(frames-V).norm(dim=(-1,-2))/2
    gap=(proj@proj.transpose(-2,-1)-V@V.T).norm(dim=(-2,-1))/2
    rates=(counts/len(q)).tolist()
    return [dict(seed=seed,n=n,readout=NAMES[j%4],optimizer=METHODS[j%4],
                 deployment="native" if j<4 else "polar",error=v[0],hop1=v[1],hop2=v[2],
                 hop3=v[3],hop4=v[4],complete_path=v[5],frame=float(dif[j]),projector=float(gap[j]))
            for j,v in enumerate(rates)]


def run(seed):
    V,R=torch.linalg.qr(torch.randn(31,4,device=DEV,generator=rng(seed,"frame")))
    V=V*torch.sign(torch.diag(R))
    U=torch.zeros(4,31,4,device=DEV,requires_grad=True)
    first=probe(seed,V,U)
    test=draw(rng(seed,"evaluation"),V,P["test"])
    train=rng(seed,"training")
    m,v=torch.zeros_like(U),torch.zeros_like(U)
    rows=[]
    parameters={"teacher":V.cpu().numpy()}
    for step in range(1,max(P["n"])//P["batch"]+1):
        g=gradient(U,draw(train,V,P["batch"]))
        with torch.no_grad():
            g/=g.norm(dim=(-1,-2),keepdim=True).clamp_min(1)
            m.mul_(.9).add_(g,alpha=.1)
            v.mul_(.999).addcmul_(g,g,value=.001)
            dm=(m/(1-.9**step))/((v/(1-.999**step)).sqrt()+1e-8)
            factor=(1+(step-1)/128)**(-.5)
            U[:2]-=factor*P["sgd_lr"]*g[:2]
            U[2:]-=factor*P["adam_lr"]*dm[2:]
            n=step*P["batch"]
            if n in P["n"]:
                rows.extend(score(seed,n,U,V,test))
                parameters[f"U_{n}"]=U.cpu().numpy().copy()
    np.savez_compressed(OUT/f"frames_{seed}.npz",**parameters)
    return rows,first


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/"registration.json").exists():
        raise FileExistsError("Use a new working directory to refit.")
    reg=dict(protocol_sha256=digest(ROOT/"protocol.json"),code_sha256=digest(ROOT/"matched.py"),
             torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name())
    (OUT/"registration.json").write_text(json.dumps(reg,indent=2))
    rows,grads=[],[]
    start=time.perf_counter()
    for seed in P["seeds"]:
        r,g=run(seed);rows.extend(r);grads.extend(g)
        print(seed,len(r),round(time.perf_counter()-start,2),flush=True)
    write("measurements.csv",rows)
    write("gradients.csv",grads)
    (OUT/"completion.json").write_text(json.dumps(dict(seeds=len(P["seeds"]),rows=len(rows),
        gradient_rows=len(grads),seconds=time.perf_counter()-start,
        evaluation_sha256=digest(OUT/"measurements.csv"),gradients_sha256=digest(OUT/"gradients.csv")),indent=2))


if __name__=="__main__":
    main()
