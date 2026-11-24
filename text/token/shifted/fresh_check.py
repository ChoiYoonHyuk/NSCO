"""Score all frozen endpoint models in NumPy on new official token strings and fresh signs."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import csv,sys,json,hashlib,subprocess,statistics,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent; HOP=ROOT.parent; U=HOP/'upstream'
sys.path[:0]=[str(U/'src')]
import numpy as np
import torch
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def ci(x):
    v=statistics.mean(x);d=2.776445105*statistics.stdev(x)/math.sqrt(len(x))
    return [v,v-d,v+d]
def read(v,g):
    if g=='parity':return v.prod(1)
    if g=='majority':return np.sign(v.sum(1))
    return v[:,0] if g=='first' else v[:,-1]
def main():
    torch.set_num_threads(2)
    assert os.getenv('PYTHONHASHSEED')=='0'
    p=json.loads((ROOT/'holdout_protocol.json').read_text())
    fitted=json.loads((ROOT/'audit.json').read_text())['frames']
    assert subprocess.check_output(['git','-C',str(U),'rev-parse','HEAD']).decode().strip()==p['pinned_commit']
    src=subprocess.check_output(['git','-C',str(U),'show',p['pinned_commit']+':src/tasks.py'])
    scope={};exec(compile(src,'pinned:src/tasks.py','exec'),scope);T=scope['InductionHopsFullSequenceTask']
    prior=json.loads((ROOT/'protocol.json').read_text())
    cfgs=[(align,g) for align in prior['alignment'] for g in prior['readouts']]
    output=[];sums={k:np.zeros(5,dtype=int) for k in cfgs};hashes=[]
    for s,st,sg in zip(p['frozen_fit_seeds'], p['fresh_upstream_streams'],p['fresh_query_sign_streams']):
        f=ROOT/f'fits_{s}.npz';assert sha(f)==next(z['frames_sha256'] for z in fitted if z['seed']==s)
        with np.load(f) as saved:Q,K,decay=(saved[k].copy() for k in ['query','key','decay'])
        T0=T(seq_len=100,char_tokens=4,min_hops=0,max_hops=16,rng=np.random.RandomState(st))
        rng=np.random.RandomState(sg);arr=[]
        for _ in range(p['examples_per_fitted_seed']//p['upstream_batch']):
            ids,y,_,_,_,meta=T0.get_batch(p['upstream_batch'],hops=3,metadata=True)
            tok=ids.numpy()[:,1:]-T0.induction_tokens
            pointer=np.array([z[1][:98] for z in meta])
            for i,xx in enumerate(tok):
                last={};first=[]
                for ix,c in enumerate(xx):
                    first.append(last.get(int(c),-2)+1 if int(c) in last else -1);last[int(c)]=ix
                assert first==pointer[i].tolist()
            L=np.stack([np.array([z[h][:98] for z in meta]) for h in range(1,4)],axis=-1)
            for h in (1,2):
                old=L[:,:,h-1];new=np.where(old<0,-1,np.take_along_axis(pointer,np.maximum(0,old),axis=1))
                assert np.array_equal(new,L[:,:,h])
            arr.append((tok,L))
        x=np.concatenate([z[0] for z in arr]);L=np.concatenate([z[1] for z in arr]);b=np.arange(len(x))
        assert len(x)==2048 and np.all(np.any(L[:,:,-1]>=0,axis=1))
        q=np.where(L[:,:,-1]>=0,rng.rand(*x.shape),-1).argmax(1)
        path=L[b,q];signs=rng.randint(2,size=x.shape)*2-1;vv=signs[b[:,None],path]
        hashes.append(dict(fitted_seed=s,upstream_seed=st,sign_seed=sg,fit_sha256=sha(f),
            token_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            query_and_sign_sha256=hashlib.sha256(q.tobytes()+signs.tobytes()).hexdigest()))
        for i,(align,g) in enumerate(cfgs):
            match=(Q[i]@K[i].T)/2;lam=np.logaddexp(0,decay[i])
            cand=np.arange(1,98);current=q;pred=[]
            for _ in range(3):
                potential=match[x[b,current,None],x[:,cand-int(align=='successor')]]
                w=potential-lam*np.log1p(np.maximum(0,current[:,None]-cand))
                w[cand>=current[:,None]]=-np.inf
                current=np.where(current<2,0,1+w.argmax(1));pred.append(current)
            pred=np.stack(pred,axis=-1);hit=pred==path;out=read(signs[b[:,None],pred],g)
            vals=np.array([np.sum(out!=read(vv,g)),hit.all(1).sum(),*hit.sum(0)]);sums[align,g]+=vals
            output.append(dict(fitted_seed=s,fresh_seq_seed=st,alignment=align,readout=g,examples=len(q),
                               **{n:int(v) for n,v in zip(p['metrics'],vals)}))
    with (ROOT/'fresh_counts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    aggregates=[]
    for (align,g),values in sums.items():
        counts={n:int(v) for n,v in zip(p['metrics'],values)}
        d=[x for x in output if x['alignment']==align and x['readout']==g]
        aggregates.append(dict(alignment=align,readout=g,examples=sum(x['examples'] for x in d),**counts,
            error_pct_seed_t_ci=ci([100*x['hard_answer_errors']/x['examples'] for x in d]),
            path_pct_seed_t_ci=ci([100*x['whole_path_correct']/x['examples'] for x in d])))
    report=dict(data='10,240 distinct fresh sequence draws shared across arms/readouts; 2,048 for each fitted seed',source_sha256=sha(Path(__file__)),
                frozen_protocol_sha256=sha(ROOT/'holdout_protocol.json'),per_seed_csv_sha256=sha(ROOT/'fresh_counts.csv'),
                fit_and_evaluation_hashes=hashes,aggregates=aggregates)
    (ROOT/'fresh_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in aggregates:print(r['alignment'],r['readout'],r['hard_answer_errors'],r['whole_path_correct'],r['examples'],flush=True)
if __name__=='__main__':main()

