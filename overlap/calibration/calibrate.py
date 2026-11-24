"""Train-only selection of Gaussian signed-hop overlap corrections in intrinsic dimension two."""
import os
for t in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[t]='2'
try:
    import resource
except ImportError:
    resource = None
from pathlib import Path
import sys,math,time,json,csv,hashlib,platform
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'grid'))
sys.path.insert(0,str(ROOT.parent/'depth'))
import numpy as np
from run_overlap import path,target,rng,filters,write,sign
from depth import draw,basis,latent_path,projected_path,gate,fit
P=json.loads((ROOT/'protocol.json').read_text())
K,D=P['depth'],P['ambient']
RHO=P['overlaps'];NGRID=P['norm_grid'];AGRID=P['direction_grid_degrees']
NORMS=sorted(set(NGRID+RHO+[r*P['short_factor'] for r in RHO]))
CF={r:i for i,r in enumerate(NORMS)}
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def rot(deg):
    a=np.deg2rad(deg)
    return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
def dump(f,rows):
    with f.open('w',newline='') as w:
        obj=csv.DictWriter(w,fieldnames=list(rows[0]));obj.writeheader();obj.writerows(rows)
def coefficients(out,p):
    pp=dict(coefficient_examples=p['public_coefficient_examples_per_norm'],batch_size=4096,maximum_gap=K-2,
            coefficient_seed=p['coefficient_seed'],normalized_hoeffding_halfwidth=str(math.sqrt(math.log(2*len(NORMS)*9/.01)/(2*p['public_coefficient_examples_per_norm']))))
    cs={};t=time.perf_counter()
    for i,r in enumerate(NORMS):
        cs[r]=filters.coefficients(pp,2,r,i)
    write(out/'public_coefficients.json',dict(seconds=time.perf_counter()-t,rows=list(cs.values())))
    return cs

def consume(q,g,Y,R,cs):
    b=sign(g[:,1:]).mean(2);out=np.empty((len(RHO),len(NORMS),D-1,2))
    for j,r in enumerate(NORMS):
        f=gate(b,g,K,'sign_magnitude',cs[r])
        for i,y in enumerate(Y):out[i,j]=R.T@((f*y)[:,None]*q)
    return out

def full_block(seed,stream,count,B,V):
    gen=rng(seed,stream);q,g,z,nu=draw(gen,count,K,D,True)
    W=np.concatenate([g[...,None],np.concatenate([z,nu],axis=3)@B.T],axis=3)
    truth=[latent_path(q@V,g,z,r) for r in RHO]
    noise=np.where(gen.random(count)<.9,1,-1)
    Y=np.stack([zz.prod(1)*noise for _,zz in truth])
    return q,W,truth,Y

def validate(q,W,labels,M,candidates):
    # Public function: no test labels, true frames, target size or direction enter joint-grid scoring.
    frames=[fit(M[CF[r]],r,theta) for r,theta in candidates]
    n=len(q);g=W[...,0]
    scores=[]
    for start in range(0,len(frames),12):
        section=frames[start:start+12]
        Kall=(W.reshape(-1,D)@np.concatenate(section,axis=1)).reshape(n,K,2,len(section),2)
        for j in range(len(section)):
            _,v=projected_path(q,g,Kall[:,:,:,j,:])
            scores.append(int(np.count_nonzero(v.prod(1)!=labels)))
    best=min(range(len(scores)),key=scores.__getitem__)
    return candidates[best],scores
def avg_angle(a,b):return abs((a-b+180)%360-180)

def run(seed,p,cs,out):
    begin=time.perf_counter();B=basis(seed,D);angle=float(rng(seed,567).uniform(0,360));V=rot(angle)
    true_frames=[target(B[:,:2],r)@V.T for r in RHO]
    cv=p['inner_validation_count']
    qc,Wc,_,Yc=full_block(seed,739,cv,B,V); delta=(Wc[:,0,0]-Wc[:,0,1])/math.sqrt(2)
    Mc=consume(qc,Wc[...,0],Yc,delta[:,0,None]*delta[:,1:],cs)
    qt,Wt,truth,_=full_block(seed,859,p['test_count'],B,V)
    for i,f in enumerate(true_frames):
        ii,vv=path(qt,Wt,f)
        assert np.array_equal(ii,truth[i][0]) and np.array_equal(vv,truth[i][1])
    gen=rng(seed,941);M=np.zeros_like(Mc);prev=0;rows=[];cal=[];frames={};seen=0
    for n in p['budgets']:
        fitcount=n-cv
        while seen<fitcount:
            size=min(4096,fitcount-seen)
            q,g,z,nu=draw(gen,size,K,D,False)
            delta=(z[:,0,0]-z[:,0,1])/math.sqrt(2)
            R=(np.concatenate([delta,nu],axis=1)@B.T)*((g[:,0,0]-g[:,0,1])/math.sqrt(2))[:,None]
            noise=np.where(gen.random(size)<.9,1,-1)
            Y=np.stack([latent_path(q@V,g,z,r)[1].prod(1)*noise for r in RHO])
            M+=consume(q,g,Y,R,cs);seen+=size
        for i,r in enumerate(RHO):
            known=[(h,angle) for h in NGRID]
            joint=[(h,t) for h in NGRID for t in ([0.] if h==0 else AGRID)]
            select={}
            for z,cand in [('norm_given_direction',known),('joint_grid',joint)]:
                tic=time.perf_counter();best,counts=validate(qc,Wc,Yc[i],M[i]/fitcount,cand)
                select[z]=(best, min(counts), time.perf_counter()-tic)
                for (h,t),err in zip(cand,counts):
                    cal.append(dict(seed=seed,n=n,true_rho=r,arm=z,candidate_norm=h,candidate_angle=t,
                                    noisy_validation_errors=err,validation_examples=cv))
            names=[('oracle',(r,angle)),('short_given_direction',(r*p['short_factor'],angle))]
            names+=[(z,select[z][0]) for z in ['norm_given_direction','joint_grid']]
            for z,(h,t) in names:
                F=fit((M[i,CF[h]]+Mc[i,CF[h]])/n,h,t)
                ix,val=path(qt,Wt,F);it,vt=truth[i]
                wrong=int(np.sum(val.prod(1)!=vt.prod(1)));matches=(ix==it)
                row=dict(seed=seed,true_rho=r,true_degrees=angle,n=n,arm=z,chosen_rho=h,chosen_degrees=t,
                    angular_error_degrees=avg_angle(t,angle) if h>0 else '',norm_abs_error=abs(h-r),
                    fit_labels=fitcount,validation_labels=cv,unique_final_labels=n,
                    final_refit_labels=n,additional_supervision_labels=0,
                    noisy_validation_errors=select[z][1] if z in select else '',
                    validation_candidates=len(known) if z=='norm_given_direction' else len(joint) if z=='joint_grid' else 0,
                    validation_seconds=select[z][2] if z in select else 0.,
                    clean_test_errors=wrong,test_count=len(qt),clean_error=wrong/len(qt),
                    normalized_frame_error=np.linalg.norm(F-true_frames[i])/math.sqrt(2),
                    full_path_correct=int(matches.all(1).sum()),full_path_accuracy=matches.all(1).mean())
                row.update({'hop'+str(j+1):matches[:,j].mean() for j in range(K)})
                rows.append(row)
                if n==p['budgets'][-1]:frames[f'r{i}_{z}']=F
        print('seed',seed,'n',n,'elapsed',round(time.perf_counter()-begin,2),flush=True)
    dump(out/f'outcomes_{seed}.csv',rows);dump(out/f'selection_{seed}.csv',cal)
    np.savez_compressed(out/f'fits_{seed}.npz',**frames)
    return len(rows)
def main():
    stage=sys.argv[1];assert stage in ('smoke','production');out=ROOT/stage;out.mkdir(exist_ok=True)
    p=P.copy()
    if stage=='smoke':p.update(seeds=[81133],budgets=[8192,16384],test_count=1024,inner_validation_count=128,public_coefficient_examples_per_norm=4096)
    paths = ['protocol.json', 'calibrate.py', '../depth/depth.py', '../grid/run_overlap.py', '../grid/filters.py', '../known/scripts/first_sign.py']
    source = {name: sha(ROOT/name) for name in paths}
    write(out/'registration.json',dict(protocol=p,sha256=source,hidden_directions='Independent U[0,360); stream 567',streams=dict(fit=941,validation=739,test=859)))
    t=time.perf_counter();cs=coefficients(out,p)
    total=sum(run(s,p,cs,out) for s in p['seeds'])
    assert all(sha(ROOT/f)==h for f,h in source.items())
    write(out/'completion.json',dict(rows=total,seconds=time.perf_counter()-t,threads=2,epochs=0,gpu=False,
          peak_rss_mib=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 if resource else None),python=platform.python_version(),numpy=np.__version__,
          output_sha256={f.name:sha(f) for f in list(out.glob('*.csv'))+list(out.glob('*.npz'))}))
if __name__=='__main__':main()

