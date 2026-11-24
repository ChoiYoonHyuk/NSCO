"""Matched raw-endpoint error contrasts, with both supervised arms unshrunk."""
import json
import numpy as np
from audit import direct_replay
from prepare import ROOT, save_json

def main():
    outcome=json.loads((ROOT/'evaluated.json').read_text())
    lock=json.loads((ROOT/'selection_lock.json').read_text())
    cfg=json.loads((ROOT/'protocol.json').read_text())
    result={}
    for task,cell in outcome['results'].items():
        k=cell['k']
        a=np.load(ROOT/f'test_{k}.private.npz')
        x,e,g,n=(a[z] for z in ('f','e','g','n'))
        choice=lock['choices'][task]
        risks={}
        for iseed,seed in enumerate(cfg['seeds']):
            opts=choice['seeds'][str(seed)]['arms']
            ws={arm:np.array(r['raw_weights'],dtype=np.float32) for arm,r in opts.items()}
            ws['frozen']=np.array(cfg['frozen_configs'][choice['frozen']],dtype=np.float32)
            for arm,w in ws.items():
                path=direct_replay(x,e,n,w,k)
                r=(path[:,:,None]==g[:,None,:]).any(-1).sum(-1)
                tag=task.rstrip('3')
                if tag=='first':risk=.5*(path[:,0]!=g[:,0])
                if tag=='last':risk=.5*(path[:,-1]!=g[:,-1])
                if tag=='parity':risk=.5*(r!=k)
                if tag=='majority':risk=.5*(1-r/4-(r==3)/4)
                m=arm if arm=='frozen' else arm+'_unshrunk'
                assert abs(risk.mean()-cell['models'][m]['error']['seed_values'][iseed])<1e-9
                risks.setdefault(arm,[]).append(risk)
        out={}
        for arm,ref in [('final','frozen'),('first_value','final'),('first_index','final'),('first_index','first_value'),('all_indices','final')]:
            diff=np.array(risks[arm])-np.array(risks[ref])
            idx=np.random.default_rng(378129).integers(len(g),size=(4000,len(g)))
            out[arm+'_vs_'+ref]={'mean':float(diff.mean()),'seed_sd':float(diff.mean(-1).std(ddof=1)),
                                'question_ci95':np.quantile(diff.mean(0)[idx].mean(-1),[.025,.975]).tolist()}
        result[task]=out
    save_json('raw_pairs.json',result)

if __name__=='__main__':main()
