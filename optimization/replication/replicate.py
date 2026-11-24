"""Locked independent depth-12 replication. See plan.json for endpoints and caps."""
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import torch
import gaussian as base
import deep_adam as previous

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
PLAN = json.loads((ROOT/"plan.json").read_text())


def gradient(w, active, noisy, data, flips):
    q, signs, keys, _, c1, _, y = data
    pred, first, _ = previous.parity(w,q,signs,keys,True)
    target = torch.where(noisy[:,None],c1*flips,c1)
    loss = (pred-y).square() + active[:,None]*(first-target).square()
    return torch.autograd.grad(.5*loss.mean(1).sum(),w)[0]


def validate():
    d6 = previous.validate_against_six_hop()
    seed=708719
    w,_=base.configs(seed,[(2,a,1,1) for a in (0,1,1)])
    data=previous.draw(256,12,base.generator(seed,3),base.frame(seed))
    active=torch.tensor([False,True,True],device=base.DEVICE)
    noisy=torch.tensor([False,False,True],device=base.DEVICE)
    one=torch.ones_like(data[4])
    expected=previous.gradient(w,active,data)
    actual=gradient(w,active,noisy,data,one)
    torch.testing.assert_close(expected,actual,rtol=2e-5,atol=1e-7)
    flipped=gradient(w,active,noisy,data,-one)
    changed=tuple(-x if i==4 else x for i,x in enumerate(data))
    torch.testing.assert_close(flipped[:2],expected[:2],rtol=2e-5,atol=1e-7)
    torch.testing.assert_close(flipped[2],previous.gradient(w,active,changed)[2],rtol=2e-5,atol=1e-7)
    return d6,(actual-expected).abs().max().item()


def fit(seed):
    configs=[(r,a) for r in PLAN["rates"] for a in PLAN["arms"]]
    w,_=base.configs(seed,[(2,int(a!="final"),1,1) for r,a in configs])
    teacher=base.frame(seed)
    test=previous.draw(PLAN["test_examples"],12,base.generator(seed,5),teacher)
    train=base.generator(seed,3)
    fliprng=base.generator(seed,19)
    rate=torch.tensor([r for r,a in configs],device=base.DEVICE)[:,None,None]
    active=torch.tensor([a!="final" for r,a in configs],device=base.DEVICE)
    noisy=torch.tensor([a=="noisy" for r,a in configs],device=base.DEVICE)
    m,v=torch.zeros_like(w),torch.zeros_like(w)
    grid=set(PLAN["grid_examples"])
    batch=PLAN["batch"]
    rows=[]
    for step in range(1,PLAN["max_examples"]["final"]//batch+1):
        n=step*batch
        if n-batch==PLAN["max_examples"]["clean"]:
            keep=~active
            w=w[keep].detach().requires_grad_(True)
            rate,m,v,active,noisy=[x[keep] for x in (rate,m,v,active,noisy)]
            configs=[(r,a) for r,a in configs if a=="final"]
        data=previous.draw(batch,12,train,teacher)
        flips=(torch.rand(batch,generator=fliprng,device=base.DEVICE)>=.1).float()*2-1
        g=gradient(w,active,noisy,data,flips)
        with torch.no_grad():
            g/=g.norm(dim=(-2,-1),keepdim=True).clamp_min(1.)
            m.mul_(.9).add_(g,alpha=.1)
            v.mul_(.999).addcmul_(g,g,value=.001)
            direction=(m/(1-.9**step))/((v/(1-.999**step)).sqrt()+1e-8)
            w-=(rate/math.sqrt(1+(step-1)/128.))*direction
        if n in grid:
            errors,hits=previous.evaluate(w,test)
            for (r,a),e,h in zip(configs,errors,hits):
                rows.append(dict(seed=seed,depth=12,rate=r,arm=a,n=n,labels=n*(1+int(a!="final")),
                                 errors=e,path_hits=h,test=PLAN["test_examples"]))
    return rows


def main():
    d6,d12=validate()
    prov=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
              six_hop_gradient_max_abs_diff=d6,depth12_gradient_max_abs_diff=d12)
    for name in ["plan.json","replicate.py","gaussian.py","deep_adam.py","analyze.py"]:
        prov[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (OUT/"provenance.json").write_text(json.dumps(prov,indent=2))
    for seed in PLAN["seeds"]:
        dest=OUT/f"metrics_{seed}.csv"
        if dest.exists():
            continue
        start=time.monotonic()
        rows=fit(seed)
        temp=dest.with_suffix(".tmp")
        with temp.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(dest)
        print(f"finished {seed} in {time.monotonic()-start:.1f}s",flush=True)
    print("complete",flush=True)


if __name__=="__main__":
    main()
