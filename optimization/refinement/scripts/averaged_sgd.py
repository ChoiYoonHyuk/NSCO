"""Matched averaged-SGD refinements and independent validation protection.

The protocol and source lock are required before training. Old observations
are never overwritten. Numerical equivalence is audited independently.
"""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'2')
import argparse,hashlib,json,math,time
import numpy as np
from scipy.stats import binom
import sample_split as base
from gaussian_kernels import configure_torch,hard
from predictable_control import controlled_sums,condition_mask,predictable_polar

R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);assert not p.exists(),f'Preserve completed artifact: {p}'
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def guard(a,threshold):
    q,_=predictable_polar(a);eligible,ratio=condition_mask(a,threshold)
    return q*eligible[...,None,None],eligible,ratio
def rate(t,n,p,rate,policy):
    lr=p['step_base']*rate*(1+t/256)**(-.75)
    return lr*(.25 if policy=='half_damp' and t*p['batch_size']>=n//2 else 1.)
def clipped_numpy(g,policy):
    return g*min(1.,1./np.linalg.norm(g)) if policy=='clip1' and np.linalg.norm(g)>0 else g
def candidate_frames(U,average,A,Z,Zmean,n,mu,p):
    Q=base.polar(average);E=base.polar(U)
    moments={'plain':A/(n*p['lambda']*mu),'online':Z/(n*p['lambda']*mu),'meancontrol':Zmean/(n*p['lambda']*mu)}
    specs=[{'family':'averaged_sgd','alpha':0.},{'family':'endpoint_sgd','alpha':0.}];mats=[Q,E]
    for family,M in moments.items():
        for alpha in p['alpha_grid']:
            specs.append({'family':'avg_'+family,'alpha':alpha})
            mats.append(Q.copy() if alpha==0 else base.polar((1-alpha)*Q+alpha*M))
    return specs,np.stack(mats),moments
def hard_predictions(mats,q,W,name,p,device):
    v=base.predictions(mats,(q,W),p,device)
    return np.stack([base.gvalue(x,name) for x in v]).astype(np.int8)
def gate_indices(spec,pred,noisy,delta):
    candidates=[i for i,s in enumerate(spec) if s['family'].startswith('avg_') and s['alpha']!=0]
    baseline=pred[0];records=[];accepted=[]
    for i in candidates:
        disagreement=pred[i]!=baseline;m=int(disagreement.sum());w=int(((pred[i]==noisy)&disagreement).sum())
        pv=float(binom.sf(w-1,m,.5)) if m else 1.
        okay=pv<=delta/len(candidates)
        records.append({'candidate_index':i,'disagreements':m,'wins':w,'p_one_sided':pv,'accepted':okay})
        if okay:accepted.append(i)
    index=min(accepted,key=lambda i:(float(np.mean(pred[i]!=noisy)),abs(spec[i]['alpha']),spec[i]['alpha'],i)) if accepted else 0
    return index,records
def samples(seed,stream,k,count,P,p):
    q,W=base.draw(base.rng(seed,stream,k),count,k,p['D'],p['d'])
    _,values=hard(q,W,P)
    return q,W,values
def train_one(p,case,cfg,stage,seed,device,out):
    import torch
    started=time.monotonic();k=case['depth'];n=case['n'];bs=p['batch_size'];name=cfg['readout'];policy=cfg['policy']
    target=out/'raw'/f"{cfg['id']}_seed{seed}"
    assert not target.with_suffix('.json').exists()
    target.parent.mkdir(parents=True,exist_ok=True)
    P=base.target(seed,p);rng=base.rng(seed,p['streams']['training'],k)
    m,d=p['D']-1,p['d'];mu=base.masses(name,k)[0]/(d*math.sqrt(math.pi))
    U=torch.zeros((1,m,d),device=device,dtype=torch.float64,requires_grad=True)
    A=np.zeros((m,d));Z=A.copy();Zmean=A.copy();mean_sum=A.copy();mean_count=0
    sum_post=A.copy();count_post=0;var=np.zeros(3);invalid=np.zeros(2,dtype=np.int64)
    stream=[];label_hash=hashlib.sha256();fixtures={};checkpoints={};clipped=0
    for t in range(n//bs):
        q,W=base.draw(rng,bs,k,p['D'],d);noise=np.where(rng.random(bs)<(1+p['lambda'])/2,1.,-1.)
        _,values=hard(q,W,P);Y=base.gvalue(values,name)*noise
        stream.append(hashlib.sha256(q.tobytes()+W.tobytes()+noise.tobytes()).hexdigest());label_hash.update(Y.astype(np.int8).tobytes())
        before=U.detach().cpu().numpy()[0].copy()
        # Both controls are measurable before this batch. After the midpoint,
        # the second control averages only preceding second-half iterates.
        q0,active0,_=guard(before[None,None],p['condition_ratio_threshold'])
        previous_mean=mean_sum/mean_count if mean_count else np.zeros_like(before)
        q1,active1,_=guard(previous_mean[None,None],p['condition_ratio_threshold'])
        controls=np.concatenate((q0,q1),axis=0);active=np.concatenate((active0,active1),axis=0)
        a,_,z,diag=controlled_sums(q,W,Y[None],[name],controls,active,np.array([mu]),p['lambda'],P[1:])
        A+=a[0];Z+=z[0,0];Zmean+=z[1,0]
        var[0]+=diag[0][0];var[1:]+=diag[1][:,0];invalid+=~active[:,0]
        f=base.soft_values(U,q,W,[name],p,device)[0]
        gradient=torch.autograd.grad(.5*(f-torch.as_tensor(Y,device=device)).square().mean(),U)[0]
        assert torch.isfinite(gradient).all(), 'Nonfinite gradient retained as failure'
        norm=float(torch.linalg.vector_norm(gradient))
        effective=gradient*min(1.,1./norm) if policy=='clip1' and norm>0 else gradient
        clipped+=int(policy=='clip1' and norm>1)
        eta=rate(t,n,p,cfg['rate'],policy)
        if t in [0,n//bs//2-1,n//bs//2,3*n//bs//4-1,n//bs-1]:
            fixtures.update({f'q_{t}':q,f'W_{t}':W,f'Y_{t}':Y,f'U_{t}':before,f'grad_{t}':gradient.detach().cpu().numpy()[0],f'eta_{t}':np.array(eta)})
        with torch.no_grad():U.sub_(eta*effective)
        post=U.detach().cpu().numpy()[0]
        # Prefix average before midpoint; reset before any second-half batch.
        if (t+1)*bs==n//2:mean_sum.fill(0);mean_count=0
        else:mean_sum+=post;mean_count+=1
        if (t+1)*bs>n//2:sum_post+=post;count_post+=1
        if (t+1)*bs in [n//4,n//2,3*n//4,n]:
            checkpoints[f'U_n{(t+1)*bs}']=post.copy();checkpoints[f'Z_n{(t+1)*bs}']=Z.copy();checkpoints[f'M_n{(t+1)*bs}']=Zmean.copy()
    final=U.detach().cpu().numpy()[0].copy();average=sum_post/count_post
    specs,mats,moments=candidate_frames(final,average,A,Z,Zmean,n,mu,p)
    vq,vW,vvalues=samples(seed,p['streams']['validation'],k,p['validation_n'],P,p)
    vclean=base.gvalue(vvalues,name).astype(np.int8);ng=base.rng(seed,p['streams']['validation']+100,k)
    vnoisy=(vclean*np.where(ng.random(len(vq))<(1+p['lambda'])/2,1,-1)).astype(np.int8)
    vp=hard_predictions(mats,vq,vW,name,p,device)
    gi,grecords=gate_indices(specs,vp,vnoisy,p['gate_delta'])
    validation_sha=hashlib.sha256(vq.tobytes()+vW.tobytes()+vclean.tobytes()+vnoisy.tobytes()).hexdigest()
    selected=list(range(len(specs)))
    if stage=='confirmation':
        choices=read(R/p['selection_file'])['choices'][cfg['id']]
        selected=[0,1]+[next(i for i,s in enumerate(specs) if s['family']==family and s['alpha']==choices[family]['alpha']) for family in p['correction_families']]
    if stage=='confirmation':
        eq,eW,ev=samples(seed,p['streams']['test'],k,p['test_n'],P,p);clean=base.gvalue(ev,name).astype(np.int8)
        evaluated=selected+[gi];ep=hard_predictions(mats[evaluated],eq,eW,name,p,device)
    else:eq,eW,clean=vq,vW,vclean;evaluated=selected+[gi];ep=vp[evaluated]
    rows=[]
    for z,i in enumerate(evaluated):
        s=dict(specs[i]);isgate=z==len(evaluated)-1
        if isgate:s={'family':'gated','alpha':specs[i]['alpha'],'selected_family':specs[i]['family'],'selected_index':i}
        mat=mats[i];gain=float(np.linalg.norm(average)/math.sqrt(d))
        row={**s,'configuration':cfg['id'],'readout':name,'policy':policy,'rate':cfg['rate'],'seed':seed,'stage':stage,'n':n,
             'hard_error':float(np.mean(ep[z]!=clean)),'validation_noisy_error':float(np.mean(vp[i]!=vnoisy)),
             'frame_error':float(np.linalg.norm(mat-P[1:])/math.sqrt(d)),
             'average_rms_gain':gain,'control_inactive_batches':invalid.tolist(),'gradient_clipped_batches':clipped}
        rows.append(row)
    soft={}
    if stage=='confirmation':
        configs=np.stack([np.linalg.norm(average)/math.sqrt(d)*mats[i] for i in evaluated]+[average,final])
        outputs=[]
        with torch.no_grad():
            for i in range(0,len(eq),256):outputs.append(base.soft_values(configs,eq[i:i+256],eW[i:i+256],[name]*len(configs),p,device).cpu().numpy())
        outputs=np.concatenate(outputs,axis=1)
        for i,row in enumerate(rows):row['gain_aligned_soft_risk']=float(.5*np.mean((outputs[i]-p['lambda']*clean)**2))
        soft={'outputs':outputs,'native_average_risk':float(.5*np.mean((outputs[-2]-p['lambda']*clean)**2)),'native_endpoint_risk':float(.5*np.mean((outputs[-1]-p['lambda']*clean)**2))}
    geometry={}
    for key,mo in moments.items():
        ea=base.polar(average)-P[1:];ee=base.polar(final)-P[1:];em=mo-P[1:]
        geometry[key]={'moment_squared_error':float(np.sum(em**2)),'average_squared_error':float(np.sum(ea**2)),
                       'endpoint_squared_error':float(np.sum(ee**2)),'average_cross_error':float(np.sum(ea*em)),
                       'endpoint_cross_error':float(np.sum(ee*em))}
    np.savez_compressed(str(target)+'_states.npz',P=P,Ufinal=final,average=average,Asum=A,Zsum=Z,Zmean=Zmean,mu=np.array(mu),mats=mats,**checkpoints)
    np.savez_compressed(str(target)+'_fixtures.npz',**fixtures)
    np.savez_compressed(str(target)+'_predictions.npz',validation_clean=vclean,validation_noisy=vnoisy,validation_pred=vp,test_clean=clean,test_pred=ep,evaluated=np.array(evaluated),**({'soft_outputs':soft['outputs']} if soft else {}))
    write(str(target)+'_meta.json',{'configuration':cfg,'case':case,'seed':seed,'candidate_specs':specs,'evaluated':evaluated,
                                  'gate_index':gi,'gate_records':grecords,'geometry':geometry,'increment_variances':(var/n).tolist(),
                                  'validation_sha256':validation_sha,'evaluation_sha256':hashlib.sha256(eq.tobytes()+eW.tobytes()+clean.tobytes()).hexdigest(),
                                  'native_average_soft_risk':soft.get('native_average_risk'),'native_endpoint_soft_risk':soft.get('native_endpoint_risk'),
                                  'seconds':time.monotonic()-started})
    write(str(target)+'_stream.json',{'batch_sha256':stream,'training_labels_sha256':label_hash.hexdigest(),'rng_final':rng.bit_generator.state})
    write(target.with_suffix('.json'),rows)
    print(json.dumps({'status':'complete','stage':stage,'configuration':cfg['id'],'seed':seed,'seconds':time.monotonic()-started}),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['confirmation']);ap.add_argument('--device',default='cuda');ap.add_argument('--seed',type=int);args=ap.parse_args()
    p=read(R/'averaged_protocol.json')
    out=R/f'results/averaged_{args.mode}';files=p['source_lock_files']+([p['selection_file']] if args.mode=='confirmation' else [])
    lock={n:sha(R/n) for n in files};out.mkdir(parents=True,exist_ok=True)
    lp=out/'execution_lock.json'
    if lp.exists():assert read(lp)==lock
    else:write(lp,lock);write(out/'registered_protocol.json',p)
    runtime=configure_torch(args.device);started=time.monotonic()
    seeds=[args.seed] if args.seed else p[args.mode+'_seeds'];assert set(seeds)<=set(p[args.mode+'_seeds'])
    for seed in seeds:
        for cfg in p['configurations']:
            path=out/'raw'/f"{cfg['id']}_seed{seed}.json"
            if path.exists():continue
            case=next(c for c in p['cases'] if c['name']==cfg['case'])
            train_one(p,case,cfg,args.mode,seed,args.device,out)
    allpaths=[out/'raw'/f"{cfg['id']}_seed{s}.json" for s in p[args.mode+'_seeds'] for cfg in p['configurations']]
    if all(x.exists() for x in allpaths) and not (out/'run_manifest.json').exists():
        rows=sum((read(x) for x in allpaths),[]);write(out/'raw_results.json',rows)
        write(out/'run_manifest.json',{'status':'complete','runtime':runtime,'seconds_this_process':time.monotonic()-started,'source_lock':lock,
                                      'output_sha256':{f.relative_to(out).as_posix():sha(f) for f in sorted(out.rglob('*')) if f.is_file()}})
if __name__=='__main__':main()
