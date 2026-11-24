"""Single-pass best-position test. Run with numpy, one BLAS thread; outputs beside this script."""
import csv
import json
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
D, d, U, H, THRESH = 8, 2, 8192, 8192, .15
DEPTHS = (4, 7, 10, 13)
GRID = [round(2**(8+i/2)) for i in range(21)]
NTEACH = 8
POSITIONS = (1, 2) # zero-based singleton positions
CHUNK = 2048

def factors(x, position):
    groups = np.delete(np.arange(x.shape[1]), position).reshape(-1, 3)
    triples = x[:, groups]
    majorities = (triples.sum(axis=2)-triples.prod(axis=2))/2
    return groups, majorities

def answer(x, pos):
    _, f = factors(x, pos)
    return x[:, pos] * f.prod(axis=1)

def derivatives(x, pos):
    groups, f = factors(x, pos)
    out = np.empty_like(x, dtype=float)
    out[:, pos] = f.prod(axis=1)
    for i, group in enumerate(groups):
        other = x[:, pos] * np.delete(f, i, axis=1).prod(axis=1)
        for h, j in enumerate(group):
            l, r = np.delete(group, h)
            out[:, j] = other * (1-x[:, l]*x[:, r]) / 2
    return out

def draw_train(rng, n, k, P, positions):
    # Source simulates observed bags, and independently uses their projected keys to generate the hidden label.
    q = rng.standard_normal((n,d))
    q /= np.linalg.norm(q, axis=1)[:,None]
    state = q
    row = np.arange(n)
    prev = None
    obs, selected, features = np.empty((n,k)), np.empty((n,k)), {}
    for j in range(k):
        z = rng.standard_normal((n,2,D-1))
        v = rng.standard_normal((n,2))
        scores = (z.reshape(-1,D-1) @ P).reshape(n,2,d)
        win = np.einsum('bd,bd->b', scores[:,1]-scores[:,0],state)>0
        state = scores[row,win.astype(int)]
        signs = np.sign(v)
        obs[:,j] = signs.mean(axis=1)
        selected[:,j] = signs[row,win.astype(int)]
        if j in positions:
            circ = .5*(v[:,0]-v[:,1])[:,None]*(z[:,0]-z[:,1])
            features[j] = (circ,q if j==0 else prev)
        prev = z[:,0]+z[:,1]
    flip = np.where(rng.random(n)<.1,-1,1)
    labels = {p:answer(selected,p)*flip for p in POSITIONS}
    return obs,features,labels

def estimate(mat,j):
    if j==0:
        v = np.linalg.svd(mat,full_matrices=False)[0][:,:d]
    else:
        v = np.linalg.eigh((mat+mat.T)/2)[1][:,-d:]
    return v @ v.T

def records():
    rows=[]
    selections=[]
    for seed in range(NTEACH):
        gen=np.random.default_rng(np.random.SeedSequence([20261122,seed,1]))
        P,R=np.linalg.qr(gen.standard_normal((D-1,d)))
        P=P*np.sign(np.diag(R))[None,:]
        incoming=gen.standard_normal((H,D-1))
        diff=gen.standard_normal((H,2,D-1))
        diff=diff[:,1]-diff[:,0]
        teacher=np.einsum('bd,bd->b',incoming@P,diff@P)>0
        for k in DEPTHS:
            t=time.perf_counter()
            rng=np.random.default_rng(np.random.SeedSequence([20261122,seed,k,2]))
            calib=np.random.default_rng(np.random.SeedSequence([20261122,seed,k,3]))
            unlabeled=(2*calib.integers(0,2,(U,k,2))-1).mean(axis=2)
            chosen={}
            for p in POSITIONS:
                kap=(derivatives(unlabeled,p)**2).mean(axis=0)
                chosen[p]=int(np.argmax(kap))
                selections.append(dict(seed=seed,k=k,rotation=p+1,chosen=chosen[p]+1,estimated_max=float(kap.max()),estimated_fixed=float(kap[1])))
            js={1,*chosen.values()}
            stats={(p,j):np.zeros((D-1,d if j==0 else D-1)) for p in POSITIONS for j in {1,chosen[p]}}
            beta=.5*(13/32)**((k-1)//3)
            scaled=round(1024/beta)
            marks=sorted(set(GRID+[scaled]))
            crossings=set()
            previous=0
            for n in marks:
                for start in range(previous,n,CHUNK):
                    obs,features,ys=draw_train(rng,min(CHUNK,n-start),k,P,js)
                    for p in POSITIONS:
                        der=derivatives(obs,p)
                        for j in {1,chosen[p]}:
                            left,right=features[j]
                            stats[p,j]+=(left*(ys[p]*der[:,j])[:,None]).T@right
                previous=n
                for p in POSITIONS:
                    for arm,j in [('selected',chosen[p]),('fixed2',1)]:
                        Q=estimate(stats[p,j],j)
                        pred=np.einsum('bd,bd->b',incoming@Q,diff)>0
                        err=float((pred!=teacher).mean())
                        rows.append(dict(seed=seed,k=k,rotation=p+1,arm=arm,chosen=j+1,n=n,grid=n in GRID,scaled=n==scaled,error=err,projector_error=float(np.linalg.norm(Q-P@P.T)/np.sqrt(2*d))))
                        if n in GRID and err<=THRESH: crossings.add((p,arm))
                if n>=max(8192,scaled) and len(crossings)==4: break
            print(f'seed {seed}, depth {k}: n={n}, {len(crossings)}/4 finite crossings, {time.perf_counter()-t:.1f}s',flush=True)
    return rows,selections

def save_csv(name,table):
    with (HERE/name).open('w',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=list(table[0])); w.writeheader(); w.writerows(table)

def main():
    t=time.perf_counter()
    rows,sels=records()
    save_csv('checkpoints.csv',rows);save_csv('selections.csv',sels)
    cross=[]
    for seed in range(NTEACH):
        for k in DEPTHS:
            for pos in POSITIONS:
                for arm in ('selected','fixed2'):
                    v=[r for r in rows if (r['seed'],r['k'],r['rotation'],r['arm'])==(seed,k,pos+1,arm)]
                    vgrid=[r for r in v if r['grid']]
                    hit=next((i for i,x in enumerate(vgrid) if x['error']<=THRESH),None)
                    status='right' if hit is None else ('left' if hit==0 else 'bracket')
                    lo=vgrid[-1]['n'] if hit is None else (0 if hit==0 else vgrid[hit-1]['n'])
                    hi=None if hit is None else vgrid[hit]['n']
                    cross.append(dict(seed=seed,k=k,rotation=pos+1,arm=arm,lower=lo,upper=hi,status=status,err8192=next(r['error'] for r in v if r['n']==8192),err_scaled=next(r['error'] for r in v if r['scaled'])))
    save_csv('crossings.csv',cross)
    def task(r):return r['arm']=='selected'
    good=all(r['status']=='bracket' for r in cross if task(r))
    out=dict(seeds=NTEACH,depths=DEPTHS,D=D,d=d,calibration=U,holdout=H,grid=GRID,threshold=THRESH,selection_correct=sum(r['rotation']==r['chosen'] for r in sels),selection_total=len(sels),censoring={s:sum(r['status']==s for r in cross) for s in ('left','bracket','right')})
    interp={}
    for r in cross:
        if r['status']!='bracket':continue
        vv=[t for t in rows if all(t[a]==r[a] for a in ('seed','k','rotation','arm'))]
        e0=next(t['error'] for t in vv if t['n']==r['lower']); e1=next(t['error'] for t in vv if t['n']==r['upper'])
        frac=(e0-THRESH)/(e0-e1)
        interp[(r['seed'],r['k'],r['rotation'],r['arm'])]=2**(np.log2(r['lower'])*(1-frac)+np.log2(r['upper'])*frac)
    boot=np.random.default_rng(11933).integers(0,NTEACH,size=(10000,NTEACH))
    def summary(vals,base=0):
        v=np.asarray(vals); z=v[boot].mean(axis=1); fn=(lambda x:2**x) if base else (lambda x:x)
        return [float(fn(v.mean())),*[float(fn(x)) for x in np.quantile(z,[.025,.975])]]
    if good:
        by_seed=[]
        brack=[]
        x=np.array([1+((k-1)//3)*np.log2(32/13) for k in DEPTHS])
        weight=(x-x.mean())/np.sum((x-x.mean())**2)
        for seed in range(NTEACH):
            def geos(k,what):
                a=[r for r in cross if r['k']==k and r['seed']==seed and task(r)]
                return np.mean([np.log2(interp[seed,k,r['rotation'],'selected'] if what=='interp' else r[what]) for r in a])
            by_seed.append(sum(weight[i]*geos(k,'interp') for i,k in enumerate(DEPTHS)))
            lows=[geos(k,'lower') for k in DEPTHS]; highs=[geos(k,'upper') for k in DEPTHS]
            brack.append((sum(w*(a if w>=0 else b) for w,a,b in zip(weight,lows,highs)),sum(w*(b if w>=0 else a) for w,a,b in zip(weight,lows,highs))))
        out['selected_slope_vs_minuslogbeta']=summary(by_seed)
        out['slope_sensitivity_all_bracket_endpoints']=np.mean(brack,axis=0).tolist()
    out['selected_depth_geomean_counts']={}
    out['selected_depth_mean_errors_n8192']={}
    out['rotated_fixed_over_selected']={}
    for k in DEPTHS:
        selected=[r for r in cross if r['k']==k and task(r)]
        means=[np.mean([r['err8192'] for r in selected if r['seed']==seed]) for seed in range(NTEACH)]
        out['selected_depth_mean_errors_n8192'][k]=summary(means)
        if all(r['status']=='bracket' for r in selected):
            out['selected_depth_geomean_counts'][k]=summary([np.mean([np.log2(interp[s,k,p+1,'selected']) for p in POSITIONS]) for s in range(NTEACH)],1)
        if all((s,k,3,arm) in interp for s in range(NTEACH) for arm in ['fixed2','selected']):
            out['rotated_fixed_over_selected'][k]=summary([np.log2(interp[s,k,3,'fixed2']/interp[s,k,3,'selected']) for s in range(NTEACH)],1)
    vals=[np.mean([r['err_scaled'] for r in cross if task(r) and r['seed']==s and r['k']==13])-np.mean([r['err_scaled'] for r in cross if task(r) and r['seed']==s and r['k']==4]) for s in range(NTEACH)]
    out['scaled_error_depth13_minus4']=summary(vals)
    out['seconds']=time.perf_counter()-t
    (HERE/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':
    main()

