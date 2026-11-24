"""Stream the observable Gaussian sufficient statistic. Source modules come from the archived run."""
import os
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="2"
import json, math, sys, time, hashlib, csv, platform
try:
    import resource
except ImportError:
    resource = None
from pathlib import Path
from functools import lru_cache
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"grid"))
from run_overlap import filters, recover, evaluate_public, target, rng, sign, METHODS, write
ROOT=Path(__file__).resolve().parent
def sha(f): return hashlib.sha256(f.read_bytes()).hexdigest()

def rowspec(p):
    for r in p['overlaps']:
        for k in p['depths']:
            for m in METHODS: yield (r,k,m,'exact',r,0)
            if k in p['perturbed_depths'] and r>0:
                yield (r,k,METHODS[-1],'direction',r,p['direction_degrees'])
                yield (r,k,METHODS[-1],'magnitude',r*p['magnitude_multiplier'],0)

def draw(gen,n,k,dim,full):
    q=gen.normal(size=(n,2));q/=np.linalg.norm(q,axis=1)[:,None]
    g=gen.normal(size=(n,k,2)); z=gen.normal(size=(n,k,2,2))
    nu=gen.normal(size=((n,k,2,dim-3) if full else (n,dim-3)))
    return q,g,z,nu

def latent_path(q,g,z,r):
    K=z.copy();K[...,0]=K[...,0]*math.sqrt(1-r*r)+r*g
    return projected_path(q,g,K)

def projected_path(q,g,K):
    n,k=g.shape[:2];idx=np.empty((n,k),np.int8);v=np.empty((n,k),np.int8)
    state=q;rr=np.arange(n)
    for h in range(k):
        gap=K[:,h,1]-K[:,h,0]
        j=(gap[:,0]*state[:,0]+gap[:,1]*state[:,1]>0).astype(np.int8)
        state=K[rr,h,j];idx[:,h]=j;v[:,h]=np.where(g[rr,h,j]>=0,1,-1)
    return idx,v

def fit(M,r,theta):
    angle=math.radians(theta);c=math.cos(angle);s=math.sin(angle)
    V=np.array([[c,-s],[s,c]])
    return recover(M@V,r,2)@V.T

def gate(b,g,k,m,C):
    b=b[:,:k-1]; g=g[:,1:k]
    if m==METHODS[0]:return b.prod(1)
    eta=[u['projected_estimate'] for u in C['coefficients']]
    if m==METHODS[1]:return filters.gate(b,eta,0,False)
    if m==METHODS[2]:return filters.gate(b,eta,k-2,True)
    mag=C['magnitude'];psi=(b==0)*((np.abs(g).sum(2)>=2)-mag['tail_probability'])/mag['normalizer']
    return filters.gate(b,eta,k-2,True,psi,mag['estimate'])

def basis(seed,D):
    b,R=np.linalg.qr(rng(seed,D+1000).normal(size=(D-1,D-1)))
    return b*np.sign(np.diag(R))[None,:]

def test_data(p,seed):
    k=max(p['depths']);Dmax=max(p['ambient'])
    q,g,z,nu=draw(rng(seed,202),p['test_examples'],k,Dmax,True)
    X=np.concatenate([z,nu],axis=3)
    truths={r:latent_path(q,g,z,r) for r in p['overlaps']}
    return q,g,X,truths

def seed_run(p,seed,cs,out):
    specs=list(rowspec(p));S=len(specs);kmax=max(p['depths']);Ds=p['ambient'];Dmax=max(Ds)
    bases={D:basis(seed,D) for D in Ds};train=rng(seed,302)
    qt,gt,Xt,truths=test_data(p,seed)
    pub={(r,k):evaluate_public(qt,gt[...,None],k,cs[r],v[:,:k].prod(1)) for r,(idx,v) in truths.items() for k in p['depths']}
    moment={D:np.zeros((S,D-1,2)) for D in Ds}; signal=np.zeros(S);sq=np.zeros(S)
    ts_gate=np.zeros(S);ts_m={D:np.zeros(S) for D in Ds};ts_sim=0.
    start=time.perf_counter();seen=0;rows=[];frames={}
    for n in p['budgets']:
        while seen<n:
            tic=time.perf_counter();B=min(p['batch_size'],n-seen)
            q,g,z,nu=draw(train,B,kmax,Dmax,False)
            noise=np.where(train.random(B)<(1+p['lambda'])/2,1,-1)
            zz=(z[:,0,0]-z[:,0,1])/math.sqrt(2)
            left=(g[:,0,0]-g[:,0,1])/math.sqrt(2)
            xx=np.concatenate([zz,nu],axis=1)
            R={D:(xx[:,:D-1]@bases[D].T)*left[:,None] for D in Ds}
            selected={r:latent_path(q,g,z,r)[1] for r in p['overlaps']}
            tails={r:np.cumprod(v[:,1:],axis=1,dtype=np.int8) for r,v in selected.items()}
            b=sign(g[:,1:]).mean(2);ts_sim+=time.perf_counter()-tic
            cache={};momcache={}
            for j,(r,k,m,setting,rfit,deg) in enumerate(specs):
                key=(k,m,rfit if m!='certificate' else 0)
                if key not in cache:
                    tic=time.perf_counter();f=gate(b,g,k,m,cs[rfit])
                    cache[key]=f,time.perf_counter()-tic
                f,cost=cache[key];ts_gate[j]+=cost
                t=tails[r][:,k-2];signal[j]+=np.dot(f,t);sq[j]+=np.dot(f,f)
                key=(r,k,m,rfit)
                if key not in momcache:
                    yq=((selected[r][:,0]*noise*t)*f)[:,None]*q
                    update={}
                    for D in Ds:
                        tic=time.perf_counter();delta=R[D].T@yq
                        update[D]=delta,time.perf_counter()-tic
                    momcache[key]=update
                for D,(delta,cost) in momcache[key].items():
                    moment[D][j]+=delta;ts_m[D][j]+=cost
            seen+=B
        for j,(r,k,m,setting,rfit,deg) in enumerate(specs):
            ti,tv=truths[r];y=tv[:,:k].prod(1)
            for D in Ds:
                tic=time.perf_counter();P=fit(moment[D][j]/n,rfit,deg);fit_s=time.perf_counter()-tic
                K=Xt[:,:k,:,:D-1]@(bases[D].T@P[1:])+gt[:,:k,:,None]*P[0]
                idx,v=projected_path(qt,gt[:,:k],K);agree=(idx==ti[:,:k])
                ferror=float(np.linalg.norm(P-target(bases[D][:,:2],r))/math.sqrt(2))
                err=float(np.mean(v.prod(1)!=y))
                row=dict(seed=seed,d=2,D=D,rho=r,k=k,method=m,setting=setting,supplied_norm=rfit,
                    degrees=deg,n=n,clean_error=err,frame_error_normalized=ferror,
                    complete_chain_accuracy=agree.all(1).mean(),public_values_only_error=pub[r,k],
                    gate_second_moment=sq[j]/n,gate_identifying_signal=signal[j]/n,
                    gate_snr=(signal[j]/n)**2/(sq[j]/n),
                    gate_seconds=ts_gate[j],moment_seconds=ts_m[D][j],recovery_seconds=fit_s,
                    shared_simulation_seconds=ts_sim,coefficient_seconds=0. if m==METHODS[0] else cs[rfit]['seconds'],
                    public_mc_examples=0 if m==METHODS[0] else p['coefficient_examples'],
                    final_labels=n,additional_supervision_labels=0,test_examples=len(qt))
                row.update({'hop'+str(h+1):agree[:,h].mean() if h<k else '' for h in range(kmax)})
                rows.append(row)
                if n==p['budgets'][-1]: frames['s'+str(j)+'D'+str(D)]=P
        print('PROGRESS',seed,n,round(time.perf_counter()-start,1),flush=True)
    with (out/f'{seed}.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(out/f'frames_{seed}.npz',**frames)
    return len(rows)

def main():
    stage=sys.argv[1];out=ROOT/stage;out.mkdir(exist_ok=True)
    p=json.loads((ROOT/'protocol.json').read_text())
    if stage=='smoke':
        p.update(budgets=[4096,8192],coefficient_examples=4096,test_examples=512,seeds=[99517])
    source={f:sha(ROOT/f) for f in ['protocol.json','depth.py','../grid/run_overlap.py','../known/scripts/first_sign.py','../grid/filters.py']}
    write(out/'registration.json',{'p':p,'source_sha256':source})
    specs=list(rowspec(p));rhos=sorted(set(s[4] for s in specs))
    cp=dict(coefficient_examples=p['coefficient_examples'],batch_size=4096,maximum_gap=max(p['depths'])-2,
        coefficient_seed=p['coefficient_seed'],normalized_hoeffding_halfwidth=str(math.sqrt(math.log(2*len(rhos)*9/.01)/(2*p['coefficient_examples']))))
    cs={};begin=time.perf_counter()
    for i,r in enumerate(rhos):
        tic=time.perf_counter();c=filters.coefficients(cp,2,r,i);c['seconds']=time.perf_counter()-tic;cs[r]=c
    write(out/'coefficients.json',list(cs.values()))
    nr=sum(seed_run(p,s,cs,out) for s in p['seeds'])
    assert all(sha(ROOT/f)==h for f,h in source.items())
    write(out/'completion.json',dict(rows=nr,seconds=time.perf_counter()-begin,threads=2,cpu_only=True,
        peak_mib=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 if resource else None),
        host=platform.node(),numpy=np.__version__,results={f.name:sha(f) for f in sorted(out.glob('*.csv'))}))
if __name__=='__main__': main()
