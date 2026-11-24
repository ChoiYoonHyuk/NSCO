"""Independent NumPy hard-rollout replay, sign integration, and scorer invariance checks."""
from itertools import product, permutations
from functools import lru_cache
import json
import math
import numpy as np
from prepare import ROOT, save_json, sha

@lru_cache(None)
def signs_error(task, k, selected, truth):
    m=max(selected+truth)+1
    possible=np.array(list(product([-1,1],repeat=m)),dtype=np.int8)
    def output(order):
        values=possible[:,order]
        if task=='first':return values[:,0]
        if task=='last':return values[:,-1]
        if task=='parity':return values.prod(-1)
        return np.sign(values.sum(-1))
    return float((output(selected)!=output(truth)).mean())

def direct_replay(f,e,n,w,k):
    sf=np.einsum('ncf,f->nc',f,w[:4],optimize=False)
    edge=np.einsum('nijf,f->nij',e,w[4:],optimize=False)
    rows=np.arange(len(n))
    candidates=np.arange(f.shape[1])[None,:]
    chosen=[]
    scores=sf.copy()
    for _ in range(k):
        scores[candidates>=n[:,None]]=-np.inf
        for past in chosen:scores[rows,past]=-np.inf
        advance=scores.argmax(-1)
        chosen.append(advance)
        scores=edge[rows,advance].copy()
    return np.stack(chosen,axis=1)

def main():
    lock=json.loads((ROOT/'selection_lock.json').read_text())
    protocol=json.loads((ROOT/'protocol.json').read_text())
    outcome=json.loads((ROOT/'evaluated.json').read_text())
    assert outcome['selection_sha256']==sha(ROOT/'selection_lock.json')
    assert lock['code']==sha(ROOT/'bridge.py') and lock['features_code']==sha(ROOT/'prepare.py')
    assert lock['protocol']==sha(ROOT/'protocol.json')
    max_diff=0.
    replayed=0
    paired={}
    for task,cell in outcome['results'].items():
        k=cell['k']
        a=np.load(ROOT/f'test_{k}.private.npz')
        x,g=a['f'],a['g']
        edges,n=a['e'],a['n']
        c=lock['choices'][task]
        base=np.array(protocol['frozen_configs'][c['frozen']],np.float32)
        frac_keys=protocol['shrinkage']
        paired.setdefault(k,{})
        for iseed,seed in enumerate(protocol['seeds']):
            v=c['seeds'][str(seed)]
            assert paired[k].setdefault(seed,v['training_digest'])==v['training_digest']
            weights={'e5':np.array(protocol['frozen_configs']['e5'],np.float32),'frozen':base}
            for arm,item in v['arms'].items():
                raw=np.array(item['raw_weights'],np.float32)
                assert item['scale']==frac_keys[np.argmin(item['validation_mistakes'])]
                weights[arm]=base+(raw-base)*item['scale']
                weights[arm+'_unshrunk']=raw
            for arm,w in weights.items():
                route=direct_replay(x,edges,n,w,k)
                expected=[]
                for gt,r in zip(g.tolist(),route.tolist()):
                    # Renumber to a contiguous alphabet, then enumerate every sign assignment.
                    unique={x:i for i,x in enumerate(dict.fromkeys(gt+r))}
                    u=lambda seq:tuple(unique[x] for x in seq)
                    expected.append(signs_error(task.rstrip('3'),k,u(r),u(gt)))
                observed=cell['models'][arm]
                metric={'error':np.mean(expected),'chain':np.all(g==route,axis=1).mean(),
                        'set':np.all(np.sort(g)==np.sort(route),axis=1).mean()}
                metric.update({f'hop{i+1}':(g[:,i]==route[:,i]).mean() for i in range(k)})
                for name,val in metric.items():
                    max_diff=max(max_diff,abs(val-observed[name]['seed_values'][iseed]))
                replayed+=1
                if arm=='frozen' and iseed==0:
                    # Reverse the valid candidate order, permuting both conditional axes.
                    rev=np.arange(x.shape[1])[None,:].repeat(len(n),0)
                    for j,num in enumerate(n):rev[j,:num]=rev[j,:num][::-1]
                    rr=np.arange(len(n))[:,None]
                    xp=x[rr,rev]; ep=edges[np.arange(len(n))[:,None,None],rev[:,:,None],rev[:,None,:]]
                    again=direct_replay(xp,ep,n,w,k)
                    assert np.array_equal(rev[rr,again],route)
    assert max_diff<1e-7,max_diff
    # Check exact soft readout against enumerating ordered candidate paths (both depths).
    import torch
    from bridge import distribution,D
    rng=np.random.default_rng(3824)
    for k in (2,3):
        s=rng.normal(size=(2,5)); e=rng.normal(size=(2,5,5))
        for z in e:np.fill_diagonal(z,-1e4)
        bits=rng.choice([-1.,1.],(2,8,5))
        for task,depth in protocol['readouts'].items():
            if depth!=k:continue
            result,_=distribution(*(torch.as_tensor(z,device=D,dtype=torch.float64) for z in [s,e,bits]),task)
            expected=np.zeros((2,8))
            for b in range(2):
                for seq in permutations(range(5),k):
                    prob=1
                    for j,index in enumerate(seq):
                        prev=s[b] if j==0 else e[b,seq[j-1]]
                        prob*=np.exp(prev[index])/sum(np.exp(prev[h]) for h in range(5) if h not in seq[:j])
                    vals=bits[b][:,seq]
                    tag=task.rstrip('3')
                    y={'first':vals[:,0],'last':vals[:,-1],'parity':vals.prod(-1),'majority':np.sign(vals.sum(-1))}[tag]
                    expected[b]+=prob*y
            assert np.allclose(expected,result.cpu().numpy(),atol=1e-10)
    save_json('audit.json',{'independent_numpy_replays':replayed,'max_absolute_rate_difference':max_diff,
              'candidate_permutation_equivariance':True,'all_sign_assignments_enumerated_for_error':True,
              'soft_marginal_equals_explicit_path_enumeration':True,'verified_order_and_signs_paired_across_tasks_and_arms':True,
              'evaluated_sha256':sha(ROOT/'evaluated.json'),'lock_sha256':sha(ROOT/'selection_lock.json')})
    print('audited',replayed,max_diff,flush=True)

if __name__=='__main__':main()
