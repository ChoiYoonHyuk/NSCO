"""A zero-epoch, clean-risk depth comparison on independent binary Gaussian bags."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import csv
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path
import numpy as np

DIR=Path(__file__).resolve().parent
P={"D":8,"d":2,"depths":[3,6,9],"contrast":0.8,"test_n":16384,
   "seeds":[891041,891053,891059,891071,891079,891103,891109,891127],
   "log2_budgets":list(range(7,25)),"target":0.125,"train_chunk":8192,
   "primary_outcome":"first clean-test error <=1/8; interpolate linearly in log2(n) and log(error) between tested brackets",
   "censoring":"record all unresolved cells; no extrapolation; slopes only fully bracketed families",
   "methods":"parity and product-Maj3 first-position derivative moment; terminal split later-position/orientation; +C1 uses noisy first moment",
   "label_accounting":"n examples cost n final labels or 2n labels for final+C1",
   "comparisons":"shared target and clean test across all cells; primary families and +C1 share training input and independent auxiliary flips; separate nested terminal subspace/orientation streams"}
DEPTHS=P["depths"]

def rng(seed,stream):
    return np.random.default_rng(np.random.SeedSequence([seed,stream]))

def sign(x):
    return np.where(x>=0,1,-1)

def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-30)

def polar(x):
    a,_,b=np.linalg.svd(x,full_matrices=False)
    return a@b

def raw(gen,n):
    q=unit(gen.normal(size=(n,2)))
    g=gen.normal(size=(n,9,2))
    z=gen.normal(size=(n,9,2,7))
    return q,g,z

def path(q,g,keys,force=None):
    state=q; rows=np.arange(len(q)); outputs=[]
    for h in range(g.shape[1]):
        choice=(np.einsum("ni,ni->n",keys[:,h,1]-keys[:,h,0],state)>0).astype(int)
        if h==0 and force is not None:
            choice[:]=force
        outputs.append(sign(g[rows,h,choice]))
        state=keys[rows,h,choice]
    return np.stack(outputs,axis=1)

def majority(x):
    return (x[...,0]+x[...,1]+x[...,2]-x.prod(axis=-1))/2

def read(c,k,name):
    if name=="terminal":
        return c[:,k-1]
    if name=="parity":
        return c[:,:k].prod(axis=1)
    return majority(c[:,:k].reshape(-1,k//3,3)).prod(1)

def weight(b,k,name):
    if name=="parity":
        return b[:,1:k].prod(1)
    other=majority(b[:,3:k].reshape(len(b),(k-3)//3,3)).prod(1)
    return (1-b[:,1]*b[:,2])*other/2

def metrics(A,k,name,test):
    q,g,z,T=test
    c=path(q,g[:,:k],z[:,:k]@A)
    return float(np.mean(read(c,k,name)!=read(T,k,name)))

def record(rows,seed,k,name,mode,n,err,frame,target):
    rows.append({"seed":seed,"k":k,"family":name,"supervision":mode,"examples":n,
                 "labels":n if mode=="final" else 2*n,"clean_error":err,
                 "normalized_frame_error":float(np.linalg.norm(frame-target)/math.sqrt(2))})

def primary(seed,target,test,rows):
    gen=rng(seed,203); 
    arms=list(itertools.product(DEPTHS,("parity","majority_product"),("final","plus_C1")))
    active=set(arms)
    ms={(k,s):np.zeros((7,2)) for k,s in itertools.product(DEPTHS,("parity","majority_product"))}
    aux=np.zeros((7,2));seen=0
    for jp in P["log2_budgets"]:
        n=2**jp
        while seen<n:
            m=min(n-seen,P["train_chunk"])
            q=unit(gen.normal(size=(m,2)))
            gap=gen.normal(size=(m,7))
            g=gen.normal(size=(m,2))
            ix=(np.einsum("bi,bi->b",gap,q@target.T)<0).astype(int)
            c1=sign(g[np.arange(m),ix])
            suff=gap*((g[:,0]-g[:,1])/math.sqrt(2))[:,None]
            noise=sign(.9-gen.random(m))
            extra=sign(.9-gen.random(m))
            # Given the keys and first bag, selected and competing signs at every
            # downstream hop are independent fair bits in independent-value bags.
            records=2*gen.integers(2,size=(m,8,2),dtype=np.int8)-1
            c=np.column_stack((c1,records[:,:,0]))
            b=np.column_stack((sign(g).mean(1),records.mean(2)))
            if any(t[2]=="plus_C1" for t in active):
                aux+=suff.T@((extra*c1)[:,None]*q)
            for k,s in ms:
                if (k,s,"final") in active:
                    score=noise*read(c,k,s)*weight(b,k,s)
                    ms[k,s]+=suff.T@(score[:,None]*q)
            seen+=m
        for k,s,mode in arms:
            if (k,s,mode) not in active: continue
            A=polar(ms[k,s] if mode=="final" else aux)
            err=metrics(A,k,s,test)
            record(rows,seed,k,s,mode,n,err,A,target)
            if err<=P["target"]: active.remove((k,s,mode))
        print("primary",seed,n,len(active),flush=True)
        if not active: break

def terminal(seed,target,test,rows):
    gen=rng(seed,305);seen=0;active=set(DEPTHS)
    mats={k:np.zeros((7,7)) for k in DEPTHS}
    for jp in P["log2_budgets"]:
        n=2**jp;nhalf=n//2
        while seen<nhalf:
            count=min(nhalf-seen,2048);q,g,z=raw(gen,count)
            c=path(q,g,z@target);flip=sign(.9-gen.random(count))
            for k in active:
                h=k-1
                R=(g[:,h,0]-g[:,h,1])[:,None]*(z[:,h,0]-z[:,h,1])/2
                mats[k]+=R.T@((flip*c[:,h])[:,None]*z[:,h-1].sum(1))
            seen+=count
        bases={k:np.linalg.eigh((mats[k]+mats[k].T)/2)[1][:,-2:] for k in active}
        ori={k:np.zeros((2,2)) for k in active}
        # A counter-based block seed gives the exact same orientation prefix at
        # every tested budget, independently of subspace training observations.
        for offset in range(0,nhalf,2048):
            stream=np.random.default_rng(np.random.SeedSequence([seed,407,offset//2048]))
            qq,gg,zz=raw(stream,2048)
            epsilon=sign(.9-stream.random(2048))
            m=min(2048,nhalf-offset);qq=qq[:m];gg=gg[:m];zz=zz[:m]
            vc=path(qq,gg,zz@target)
            for k,U in bases.items():
                keys=zz[:,:k]@U
                u=unit(keys[:,0,0]-keys[:,0,1])
                v0=path(qq,gg[:,:k],keys,force=0)[:,-1]
                v1=path(qq,gg[:,:k],keys,force=1)[:,-1]
                ori[k]+=u.T@((epsilon[:m]*vc[:,k-1]*(v0-v1))[:,None]*qq)
        for k in sorted(list(active)):
            A=bases[k]@polar(ori[k])
            err=metrics(A,k,"terminal",test)
            record(rows,seed,k,"terminal","final",n,err,A,target)
            if err<=P["target"]:active.remove(k)
        print("terminal",seed,n,len(active),flush=True)
        if not active:break

def checks():
    g=rng(4,1)
    q,gg,z=raw(g,29);t=polar(g.normal(size=(7,2)))
    k=z@t
    choice=np.einsum("bnd,bd->bn",k[:,0],q).argmax(1)
    assert np.all(path(q,gg,k)[:,0]==sign(gg[np.arange(len(q)),0,choice]))
    b=g.uniform(-1,1,size=(99,9))
    for d in DEPTHS:
        for s in ("parity","majority_product"):
            plus=b.copy();minus=b.copy();plus[:,0]=1;minus[:,0]=-1
            assert np.allclose((read(plus,d,s)-read(minus,d,s))/2,weight(b,d,s))
    R0=(gg[:,0,0]-gg[:,0,1])[:,None]*(z[:,0,0]-z[:,0,1])/2
    assert np.allclose(R0,(gg[:,0,:,None]*(z[:,0]-z[:,0].mean(1,keepdims=True))).sum(1))

def run(seed):
    out=DIR/"results";out.mkdir(exist_ok=True)
    ck=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    checks();start=time.perf_counter()
    target=polar(rng(seed,100).normal(size=(7,2)))
    q,g,z=raw(rng(seed,101),P["test_n"])
    test=q,g,z,path(q,g,z@target)
    rows=[]
    terminal(seed,target,test,rows)
    primary(seed,target,test,rows)
    with (out/f"{seed}.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (out/f"{seed}.json").write_text(json.dumps({"seed":seed,"sha256":ck,"seconds":time.perf_counter()-start,"rows":len(rows)}))

if __name__=="__main__":
    if sys.argv[1]=="plan":
        checks()
        print(json.dumps({"protocol":P,"sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    else:
        run(int(sys.argv[1]))
