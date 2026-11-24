"""Single-pass value versus index supervision. Fit and freeze before held-out evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from prepare import ROOT, save_json, sha

SEEDS=[1101,2202,3303,4404,5505]
TASKS={'first':2,'last':2,'parity':2,'first3':3,'last3':3,'parity3':3,'majority':3}
ARMS=['final','first_value','first_index','all_indices']
FROZEN={
    'e5':[1,0,0,0, 0,0,0,0,1,0,0],
    'e5_bge_link':[1,.45,1,1, 0,.60,0,0,1,1.5,.25],
    'bge_link':[.15,1,0,0, 0,1,0,0,.45,1.5,.40],
    'lexical_start':[.9,.25,2.5,2.5, 0,1,0,0,.7,2,0],
}
PROTOCOL={'seeds':SEEDS,'readouts':TASKS,'arms':ARMS,'frozen_configs':FROZEN,
          'epochs':1,'train_sign_draws_per_question':8,'selection_sign_draws':64,'test_sign_check':64,
          'optimizer':'Adam, lr 0.055, batch size 32; 11 shared scalar scoring coefficients; gradient norm clip 2',
          'shrinkage':[0,.3333333333333333,1],'selection':'minimize identical validation 64-draw final-label error. First choose frozen initialization, then per-arm per-seed residual scale. Ties prefer the first frozen entry and smaller scale.',
          'loss':'Exact soft path marginal binary cross entropy. Each auxiliary adds one equally weighted mean loss: C1 BCE, I1 first CE, or mean of all teacher-forced per-hop CEs. The average keeps duplicated First/C1 data from changing gradient scales.',
          'training':'Fixed question permutation and eight independent publicly observed sign draws per question and seed. Candidate order independent of seed, fixed by question ID before labels. Only declared training labels passed to optimize().',
          'evaluation':'Greedy rollout, exact random-sign error, complete ordered path, unordered set, per-position rollout and independently sampled sign errors. Paired cluster bootstrap conditional on the fixed five seeds.',
          'known_source':'MuSiQue source and E5 caches previously analyzed. Official train for train/selection, official dev for held-out controlled question evaluation. Supplied candidates contain every gold support.'}
D='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(3)
torch.backends.cuda.matmul.allow_tf32=False

def load(part,k):
    with np.load(ROOT/f'{part}_{k}.private.npz') as f:
        return {a:torch.as_tensor(f[a],device=D) for a in f}

def subset(x,ids):
    return {a:v[ids] for a,v in x.items()}

def route(x,w):
    direct=x['f']@w[:4]
    edge=x['e']@w[4:]
    valid=torch.arange(direct.shape[1],device=D)[None,:]<x['n'][:,None]
    eye=torch.eye(direct.shape[1],dtype=torch.bool,device=D)
    return direct.masked_fill(~valid,-1e4),edge.masked_fill(~valid[:,None,:]|eye[None,:,:],-1e4)

def path(s,e,k):
    b=torch.arange(len(s),device=D)
    i=s.argmax(-1)
    j=e[b,i].argmax(-1)
    if k==2:
        return torch.stack([i,j],-1)
    last=e[b,j].clone()
    last[b,i]=-1e4
    return torch.stack([i,j,last.argmax(-1)],-1)

def signs(n,count,rng):
    return torch.tensor(rng.integers(0,2,(n,count,20),dtype=np.int8),dtype=torch.float32,device=D)*2-1

def readout(v,g,task):
    task=task.rstrip('3')
    idx=g[:,None,:].expand(-1,v.shape[1],-1)
    selected=v.gather(-1,idx)
    if task=='first':return selected[:,:,0]
    if task=='last':return selected[:,:,-1]
    if task=='parity':return selected.prod(-1)
    return selected.sum(-1).sign()

def distribution(s,e,v,task):
    k=TASKS[task]
    task=task.rstrip('3')
    a=s.softmax(-1);ab=a[:,:,None]*e.softmax(-1)
    first=torch.einsum('bi,bvi->bv',a,v)
    if task=='first': return first,first
    if k==2 and task=='last': return torch.einsum('bj,bvj->bv',ab.sum(1),v),first
    if k==2 and task=='parity': return torch.einsum('bij,bvi,bvj->bv',ab,v,v),first
    eye=torch.eye(s.shape[1],device=D,dtype=torch.bool)
    # Each new hop excludes both previously visited indices.
    third=e[:,None,:,:].masked_fill(eye[None,:,None,:],-1e4).softmax(-1)
    abc=ab[:,:,:,None]*third
    last=torch.einsum('bk,bvk->bv',abc.sum((1,2)),v)
    if task=='last':return last,first
    product=torch.einsum('bijk,bvi,bvj,bvk->bv',abc,v,v,v)
    if task=='parity':return product,first
    second=torch.einsum('bj,bvj->bv',ab.sum(1),v)
    return (first+second+last-product)*.5,first

def bce(pred,y):
    return F.binary_cross_entropy((pred*.5+.5).clamp(1e-6,1-1e-6),y*.5+.5)

def observed_labels(v,g,task,arm):
    obs={'final':readout(v,g,task)}
    if arm=='first_value':obs['c1']=readout(v,g,'first')
    if arm=='first_index':obs['i1']=g[:,0]
    if arm=='all_indices':obs['indices']=g
    return obs

def optimize(features,obs,v,order,task,w0):
    # The optimizer cannot access source questions, decomposition, or undeclared labels.
    assert set(features)=={'f','e','n'}
    w=torch.nn.Parameter(w0.clone())
    optim=torch.optim.Adam([w],lr=.055)
    for ids in order.split(32):
        s,e=route(subset(features,ids),w)
        expected,first=distribution(s,e,v[ids],task)
        loss=bce(expected,obs['final'][ids])
        if 'c1' in obs:
            loss=(loss+bce(first,obs['c1'][ids]))/2
        if 'i1' in obs:
            loss=(loss+F.cross_entropy(s,obs['i1'][ids]))/2
        if 'indices' in obs:
            g=obs['indices'][ids]
            b=torch.arange(len(g),device=D)
            losses=[F.cross_entropy(s,g[:,0]),F.cross_entropy(e[b,g[:,0]],g[:,1])]
            if g.shape[1]==3:
                third=e[b,g[:,1]].clone()
                third[b,g[:,0]]=-1e4
                losses.append(F.cross_entropy(third,g[:,2]))
            loss=(loss+torch.stack(losses).mean())/2
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w],2.)
        optim.step()
    return w.detach()

def validation_mistakes(x,w,task,v,y):
    s,e=route(x,w)
    return int((readout(v,path(s,e,TASKS[task]),task)!=y).sum())

def training_digest(order,v):
    h=hashlib.sha256(order.cpu().numpy().tobytes())
    h.update(v.to(torch.int8).cpu().numpy().tobytes())
    return h.hexdigest()

def fit():
    for name in ('selection_lock.json','evaluated.json'):
        if (ROOT/name).exists():raise FileExistsError(name)
    save_json('protocol.json',PROTOCOL)
    selections={}
    for task,k in TASKS.items():
        train=load('train',k); val=load('validation',k)
        vval=signs(len(val['g']),64,np.random.default_rng(170000+k))
        yval=readout(vval,val['g'],task)
        valx={a:val[a] for a in ('f','e','n')}
        trainx={a:train[a] for a in ('f','e','n')}
        bases={a:torch.tensor(w,dtype=torch.float32,device=D) for a,w in FROZEN.items()}
        scores={a:validation_mistakes(valx,w,task,vval,yval) for a,w in bases.items()}
        name=min(scores,key=scores.get)
        w0=bases[name]
        cell={'frozen':name,'frozen_validation_mistakes':scores,'validation_labels':yval.numel(),'seeds':{}}
        for seed in SEEDS:
            rng=np.random.default_rng(seed+13000+k)
            order=torch.tensor(rng.permutation(len(train['g'])),device=D)
            v=signs(len(order),8,rng)
            run={}
            for arm in ARMS:
                obs=observed_labels(v,train['g'],task,arm)
                w=optimize(trainx,obs,v,order,task,w0)
                validation=[validation_mistakes(valx,w0+s*(w-w0),task,vval,yval) for s in PROTOCOL['shrinkage']]
                best=int(np.argmin(validation))
                run[arm]={'raw_weights':w.tolist(),'scale':PROTOCOL['shrinkage'][best],'validation_mistakes':validation,
                          'final_observations':len(order)*8,'value_observations':len(order)*8 if arm=='first_value' else 0,
                          'index_annotations':len(order)*({'first_index':1,'all_indices':k}.get(arm,0)),
                          'gradient_updates':len(order.split(32))}
            cell['seeds'][str(seed)]={'training_digest':training_digest(order,v),'arms':run}
            print('fit',task,seed,name,flush=True)
        selections[task]=cell
    save_json('selection_lock.json',{'code':sha(__file__),'features_code':sha(ROOT/'prepare.py'),'protocol':sha(ROOT/'protocol.json'),
              'receipt':sha(ROOT/'data_receipt.json'),'choices':selections})

def public_error(n,task):
    result=0.
    for p in range(n+1):
        mean=(2*p-n)/n
        if task=='parity':
            mean=((2*p-n)**2-n)/(n*(n-1))
        if task=='parity3':
            mean=sum(((-1)**(3-i))*math.comb(p,i)*math.comb(n-p,3-i)/math.comb(n,3)
                     for i in range(4) if i<=p and 3-i<=n-p)
        if task=='majority':
            mean=sum((1 if i>=2 else -1)*math.comb(p,i)*math.comb(n-p,3-i)/math.comb(n,3)
                     for i in range(4) if i<=p and 3-i<=n-p)
        result+=math.comb(n,p)*(.5-.5*abs(mean))/(2**n)
    return result

def measurements(g,chosen,v,task):
    task=task.rstrip('3')
    hits=(chosen==g)
    overlap=(chosen[:,:,None]==g[:,None,:]).any(-1).sum(-1)
    if task=='first': exact=.5*(1-hits[:,0].float())
    if task=='last': exact=.5*(1-hits[:,-1].float())
    if task=='parity': exact=.5*(overlap!=g.shape[1]).float()
    if task=='majority': exact=.5*(1-.25*overlap-.25*(overlap==3).float())
    check=(readout(v,g,task)!=readout(v,chosen,task)).float().mean(-1)
    out={'error':exact,'chain':hits.all(-1),'set':overlap==g.shape[1],'sample_error':check}
    out.update({f'hop{j+1}':hits[:,j] for j in range(g.shape[1])})
    return {a:x.float().cpu().numpy() for a,x in out.items()}

def bootstrap(a,rng):
    n=a.shape[-1]
    ids=rng.integers(n,size=(4000,n))
    return np.quantile(a.mean(0)[ids].mean(-1),[.025,.975]).tolist()

def evaluate():
    lock=json.loads((ROOT/'selection_lock.json').read_text())
    assert lock['code']==sha(__file__) and lock['protocol']==sha(ROOT/'protocol.json') and lock['receipt']==sha(ROOT/'data_receipt.json')
    results={}
    for task,k in TASKS.items():
        choice=lock['choices'][task]
        x=load('test',k)
        g=x.pop('g')
        w0=torch.tensor(FROZEN[choice['frozen']],dtype=torch.float32,device=D)
        byarm={}
        for seed in SEEDS:
            v=signs(len(g),64,np.random.default_rng(661000+seed+k))
            weights={'e5':torch.tensor(FROZEN['e5'],dtype=torch.float32,device=D),'frozen':w0}
            for arm in ARMS:
                sel=choice['seeds'][str(seed)]['arms'][arm]
                w=torch.tensor(sel['raw_weights'],device=D)
                weights[arm]=w0+sel['scale']*(w-w0)
                weights[arm+'_unshrunk']=w
            for arm,w in weights.items():
                s,e=route(x,w)
                chosen=path(s,e,k)
                assert (chosen.sort(-1).values.diff(dim=-1)>0).all() and (chosen<x['n'][:,None]).all()
                metrics=measurements(g,chosen,v,task)
                for a,z in metrics.items():byarm.setdefault(arm,{}).setdefault(a,[]).append(z)
        summary={}
        for arm,arr in byarm.items():
            out={}
            for name,z in arr.items():
                a=np.stack(z).astype(float)
                sv=a.mean(-1)
                out[name]={'mean':float(sv.mean()),'seed_values':sv.tolist(),'seed_sd':float(sv.std(ddof=1))}
            for ref in ('frozen','final','first_value'):
                dif=np.stack(arr['error'])-np.stack(byarm[ref]['error'])
                sd=dif.mean(-1).std(ddof=1)
                out['delta_vs_'+ref]={'mean':float(dif.mean()),'seed_sd':float(sd),'seed_values':dif.mean(-1).tolist(),
                                     'question_ci95':bootstrap(dif,np.random.default_rng(378129))}
            summary[arm]=out
        results[task]={'k':k,'n':len(g),'candidate_range':[int(x['n'].min()),int(x['n'].max())],
                       'public_only_expected_error':float(np.mean([public_error(int(n),task) for n in x['n'].cpu()])),
                       'selected_frozen':choice['frozen'],'models':summary}
        print('eval',task,{a:round(summary[a]['error']['mean'],5) for a in summary},flush=True)
    save_json('evaluated.json',{'selection_sha256':sha(ROOT/'selection_lock.json'),'seeds':SEEDS,'results':results})

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['fit','evaluate'])
    globals()[parser.parse_args().action]()
