"""Observable binary first-value recovery at arbitrary value/key overlap."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_jacobi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/first_sign"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def gen(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def quadrature(d, points=128):
    x, w = roots_jacobi(points, (d-3)/2, (d-3)/2)
    return x, w / w.sum()


def scales(rho, x, w, d):
    alpha = rho*x
    m = 2/np.pi * np.arcsin(alpha / np.sqrt(2))
    b = 2*(1-alpha**2)/(np.pi*np.sqrt(2-alpha**2))
    t = float(w @ (m*x))
    hp = float(w @ (b*x*x))
    ho = (float(w @ b)-hp)/(d-1)
    return t, hp, ho


def invert(t, M, x, w):
    d = len(t)
    length = np.linalg.norm(t)
    maximum = scales(1., x, w, d)[0]
    rho = 0. if length == 0 else (1. if length >= maximum else brentq(
        lambda rr: scales(rr,x,w,d)[0]-length, 0., 1., xtol=1e-13))
    direction = t/length if length else np.zeros(d)
    r = rho*direction
    _, hp, ho = scales(rho,x,w,d)
    Hinv = np.eye(d)/ho + (1/hp-1/ho)*np.outer(direction,direction)
    estimate = M @ Hinv
    estimate[0] = r
    u, _, v = np.linalg.svd(estimate, full_matrices=False)
    return u @ v, r, hp, ho


def target(base, rho):
    P = np.vstack([np.zeros((1,base.shape[1])),base.copy()])
    P[:,0] *= np.sqrt(1-rho*rho)
    P[0,0] = rho
    return P


def query(rng, n, d):
    q = rng.standard_normal((n,d))
    return q/np.linalg.norm(q,axis=1,keepdims=True)


def hard(q, W, P):
    keys = W @ P
    state = q
    rows = np.arange(len(q))
    indices, values = [], []
    for h in range(W.shape[1]):
        index = np.einsum('bnd,bd->bn',keys[:,h],state).argmax(1)
        indices.append(index)
        values.append(np.where(W[rows,h,index,0]>=0,1,-1))
        state = keys[rows,h,index]
    return np.stack(indices,1),np.stack(values,1)


def self_test():
    d,D=4,8
    x,w=quadrature(d)
    xx,ww=quadrature(d,64)
    maximum_difference=0.
    rng=gen(81717,0)
    base,rr=np.linalg.qr(rng.standard_normal((D-1,d)))
    base *= np.where(np.diag(rr)>=0,1.,-1.)
    for rho in [0.,.25,.5,.75,.999,1.]:
        t,hp,ho=scales(rho,x,w,d)
        maximum_difference=max(maximum_difference,float(np.max(np.abs(
            np.array([t,hp,ho])-scales(rho,xx,ww,d)))))
        P=target(base,rho)
        H=np.diag([hp]+[ho]*(d-1))
        M=P@H; M[0]=0
        recovered,r,_,_=invert(np.array([t]+[0.]*(d-1)),M,x,w)
        assert np.max(np.abs(recovered-P)) < 1e-11
        assert min(hp,ho)>=np.sqrt(2)/np.pi*(d-1)/(d*(d+2))-1e-13
    assert maximum_difference < 1e-12
    # Independent ambient Monte Carlo check of both conditional identities.
    checks=[]
    for alpha in [0.,.5,.9,1.]:
        W=rng.standard_normal((250000,2,D))
        v=np.zeros(D); v[0]=alpha; v[1]=np.sqrt(1-alpha*alpha)
        idx=(W@v).argmax(1)
        C=np.where(W[np.arange(len(W)),idx,0]>=0,1.,-1.)
        delta=(W[:,0]-W[:,1])/np.sqrt(2)
        R=delta[:,0,None]*delta; R[:,0]=0
        expected_mean=2/np.pi*np.arcsin(alpha/np.sqrt(2))
        expected_R=2*(1-alpha*alpha)/(np.pi*np.sqrt(2-alpha*alpha))*v
        expected_R[0]=0
        values=np.column_stack([C,C[:,None]*R[:,1:]])
        expected=np.r_[expected_mean,expected_R[1:]]
        z=(values.mean(0)-expected)/(values.std(0,ddof=1)/np.sqrt(len(W)))
        assert np.max(np.abs(z)) < 6., z
        checks.append({'alpha':alpha,'max_standard_errors':float(np.max(np.abs(z)))})
    return {'status':'passed','quadrature64_128_max_difference':maximum_difference,
            'population_recovery_max_abs_error_bound':1e-11,'conditional_moment_checks':checks}


def run():
    cfg=json.loads((ROOT/'first_sign_protocol.json').read_text())
    lock={p:sha(ROOT/p) for p in ['first_sign_protocol.json','scripts/first_sign.py']}
    assert not (OUT/'raw_results.json').exists(), 'Refuse outcome-driven overwrite.'
    OUT.mkdir(parents=True,exist_ok=True)
    lockpath=OUT/'execution_lock.json'
    assert not lockpath.exists(), 'Execution already started: inspect incident, do not silently restart.'
    write(lockpath,lock)
    write(OUT/'self_test.json',self_test())
    D,d=cfg['D'],cfg['d']; x,w=quadrature(d)
    records=[]; started=time.monotonic()
    for seed in cfg['seeds']:
        fr=gen(seed,cfg['streams']['frame'])
        base,rr=np.linalg.qr(fr.standard_normal((D-1,d)))
        base*=np.where(np.diag(rr)>=0,1.,-1.)
        targets=[target(base,rho) for rho in cfg['rho']]
        tr=gen(seed,cfg['streams']['training'])
        te=gen(seed,cfg['streams']['test'])
        qt=query(te,cfg['test_count'],d)
        Wt=te.standard_normal((cfg['test_count'],cfg['depth_test'],2,D))
        truths=[hard(qt,Wt,P) for P in targets]
        np.savez_compressed(OUT/f'seed_{seed}_test.npz',q=qt,W=Wt,frames=targets,
                            indices=[p[0] for p in truths],values=[p[1] for p in truths])
        M=np.zeros((len(targets),D,d)); t=np.zeros((len(targets),d)); total=0
        arrays={}; rng_states={'initial':tr.bit_generator.state,'at_checkpoint':{}}
        for n in cfg['sample_counts']:
            while total<n:
                batch=min(cfg['batch'],n-total)
                q=query(tr,batch,d); W=tr.standard_normal((batch,2,D))
                noise=np.where(tr.random(batch)<(1+cfg['mu'])/2,1.,-1.)
                delta=(W[:,0]-W[:,1])/np.sqrt(2)
                R=delta[:,0,None]*delta; R[:,0]=0
                if total==0:
                    np.savez_compressed(OUT/f'seed_{seed}_fixture.npz',q=q[:128],W=W[:128],noise=noise[:128])
                for j,P in enumerate(targets):
                    idx=np.einsum('bnd,bd->bn',W@P,q).argmax(1)
                    C=np.where(W[np.arange(batch),idx,0]>=0,1.,-1.)
                    y=noise*C
                    t[j]+=np.einsum('b,bd->d',y,q)
                    M[j]+=np.einsum('b,bD,bd->Dd',y,R,q,optimize=True)
                total+=batch
            rng_states['at_checkpoint'][str(n)]=tr.bit_generator.state
            arrays[f'n{n}_M']=M.copy()/n/cfg['mu']; arrays[f'n{n}_t']=t.copy()/n/cfg['mu']
            for j,(rho,P) in enumerate(zip(cfg['rho'],targets)):
                corrected,r,hp,ho=invert(t[j]/n/cfg['mu'],M[j]/n/cfg['mu'],x,w)
                left,_,right=np.linalg.svd(M[j,1:],full_matrices=False)
                naive=np.vstack([np.zeros((1,d)),left@right])
                arrays[f'n{n}_rho{j}_r']=r
                for name,estimated in [('corrected',corrected),('orthogonal',naive)]:
                    idx,values=hard(qt,Wt,estimated)
                    frame_error=float(np.linalg.norm(estimated-P)/np.sqrt(d))
                    key=f'n{n}_rho{j}_{name}'
                    arrays[key+'_frame']=estimated; arrays[key+'_values']=values.astype(np.int8)
                    arrays[key+'_indices']=idx.astype(np.int8)
                    row={'seed':seed,'n':n,'rho':rho,'estimator':name,'frame_error':frame_error,
                         'terminal_error':float(np.mean(values[:,-1]!=truths[j][1][:,-1])),
                         'parity_error':float(np.mean(values.prod(1)!=truths[j][1].prod(1))),
                         'r_error':float(np.linalg.norm(r-P[0])),'estimated_rho':float(np.linalg.norm(r))}
                    records.append(row)
        np.savez_compressed(OUT/f'seed_{seed}_estimates.npz',**arrays)
        write(OUT/f'seed_{seed}_rng.json',rng_states)
        write(OUT/f'seed_{seed}.json',[row for row in records if row['seed']==seed])
        print(json.dumps({'seed_finished':seed,'seconds':time.monotonic()-started}),flush=True)
    assert all(sha(ROOT/p)==h for p,h in lock.items())
    write(OUT/'raw_results.json',records)
    write(OUT/'run_metadata.json',{'seconds':time.monotonic()-started,'python':__import__('sys').version,
                                 'numpy':np.__version__,'selection_performed':False,'records':len(records)})


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--run',action='store_true')
    args=ap.parse_args()
    if args.self_test: print(json.dumps(self_test()))
    elif args.run: run()
    else: ap.error('choose --self-test or --run')
