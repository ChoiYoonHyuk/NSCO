"""Fixed-grid empirical sample complexity; execute only after protocol review."""
from __future__ import annotations
import os
for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(var,'2')
import argparse,json,math,platform,time
from pathlib import Path
import numpy as np
from gaussian_model import centered_R,draw,hard_path,make_frame,noise,polar
from readout_functions import ROOT,catalogue,derivative,digest,generator,label,self_test,write_json

OUT=ROOT/'results/frame_scaling'


def lock():
    names=['scaling_settings.json','scripts/gaussian_model.py','scripts/readout_functions.py',
           'scripts/estimate_scaling.py','scripts/plot_scaling.py']
    hashes={s:digest(ROOT/s) for s in names}
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/'execution_lock.json'
    if dest.exists():assert json.loads(dest.read_text())==hashes,'Frozen source changed'
    else:
        write_json(dest,hashes)
        (OUT/'registered_protocol.json').write_bytes((ROOT/'scaling_settings.json').read_bytes())
    return hashes


def run_seed(seed,p,specs):
    dest=OUT/'seeds';dest.mkdir(exist_ok=True)
    assert not (dest/f'seed_{seed}.json').exists(),'Refuse overwrite'
    target=make_frame(generator(seed,p['streams']['frame']),p['D'],p['d'])
    qtest,gtest,ztest=draw(generator(seed,p['streams']['test']),p['test_examples'],9,p['D'],p['d'])
    itest,vtest=hard_path(qtest,gtest,ztest,target)
    np.savez_compressed(dest/f'seed_{seed}_test.npz',q=qtest,g=gtest,z=ztest,
                        target_frame=target,target_indices=itest,target_values=vtest)
    rng=generator(seed,p['streams']['training']);initial=rng.bit_generator.state
    count=len(specs);sums=np.zeros((count,p['D']-1,p['d']));aux=np.zeros_like(target)
    counts=np.zeros(count,dtype=np.int64);derivative_square=np.zeros(count);derivative_sum=np.zeros(count)
    sizes=p['sample_sizes'];chunk=p['generation_chunk'];seen=0;next_cut=0
    parameters={'target_frame':target};predictions={};states=[];records=[]
    alpha=math.sqrt(2/math.pi)/math.sqrt(math.pi)/p['d']
    for start in range(0,max(sizes),chunk):
        before=rng.bit_generator.state
        batch=min(chunk,max(sizes)-start)
        q,g,z=draw(rng,batch,9,p['D'],p['d']);indices,values=hard_path(q,g,z,target)
        zeta=noise(rng,batch,p['lambda']);xi=noise(rng,batch,p['mu'])
        annotation_uniform=rng.random((batch,8))
        annotation_noise=noise(rng,batch*8,p['mu']).reshape(batch,8)
        annotation_labels=values[:,1:]*annotation_noise
        R=centered_R(g[:,0],z[:,0])
        ds=np.column_stack([derivative(g,s,annotation_uniform,annotation_labels) for s in specs])
        ys=np.column_stack([zeta*label(values,s) for s in specs])
        weighted=ds*ys;auxlabels=xi*values[:,0]
        if start==0:
            np.savez_compressed(dest/f'seed_{seed}_raw_fixture.npz',q=q[:64],g=g[:64],z=z[:64],
                target_indices=indices[:64],target_values=values[:64],zeta=zeta[:64],xi=xi[:64],
                annotation_uniform=annotation_uniform[:64],annotation_noise=annotation_noise[:64],
                annotation_labels=annotation_labels[:64],R=R[:64],derivatives=ds[:64],labels=ys[:64])
        cursor=0
        while cursor<batch:
            stop=min(batch,cursor+sizes[next_cut]-seen);sl=slice(cursor,stop)
            sums+=np.einsum('bi,bcj->cij',R[sl],weighted[sl,:,None]*q[sl,None,:],optimize=True)
            aux+=R[sl].T@(auxlabels[sl,None]*q[sl])
            counts+=np.count_nonzero(ds[sl],axis=0);derivative_square+=(ds[sl]**2).sum(0);derivative_sum+=ds[sl].sum(0)
            seen+=stop-cursor;cursor=stop
            if seen!=sizes[next_cut]:continue
            for si,spec in enumerate([*specs,{'id':'first_value','family':'first_value','depth':1,'kappa1':1.,'beta':None}]):
                isaux=si==count;moment=aux/seen if isaux else sums[si]/seen
                kap=spec['kappa1'];signal=(p['mu'] if isaux else p['lambda'])*alpha*kap
                frame,rank=polar(moment);key=f"n{seen}_{spec['id']}"
                parameters[key+'_moment']=moment;parameters[key+'_frame']=frame
                parameters[key+'_singular']=np.linalg.svd(moment,compute_uv=False)
                ii,vv=hard_path(qtest,gtest,ztest,frame)
                predictions[key+'_indices']=ii.astype(np.uint8)
                classification=(vv[:,0]!=vtest[:,0]).mean() if isaux else (label(vv,spec)!=label(vtest,spec)).mean()
                row={'seed':seed,'n':seen,'condition':spec['id'],'family':spec['family'],'depth':spec['depth'],
                    'annotation_probability':spec.get('annotation_probability',0.),'kappa1':kap,'beta':spec['beta'],
                    'frame_error':float(np.linalg.norm(frame-target)/math.sqrt(p['d'])),
                    'moment_mse':float(np.linalg.norm(moment/signal-target)**2/p['d']),
                    'exact_moment_mse':(p['D']-1)*kap/(seen*p['d']*signal*signal)-1/seen,
                    'rank':rank,'hard_readout_error':float(classification),'path_disagreement':float((ii!=itest).any(1).mean()),
                    'nonzero_derivatives':seen if isaux else int(counts[si]),
                    'empirical_derivative_second_moment':1. if isaux else float(derivative_square[si]/seen),
                    'empirical_derivative_mean':1. if isaux else float(derivative_sum[si]/seen)}
                records.append(row)
            next_cut+=1
            print(json.dumps({'stage':'sample_scaling','seed':seed,'n':seen,'conditions':count+1}),flush=True)
            if next_cut==len(sizes):break
        states.append({'start':start,'count':batch,'rng_before':before,'rng_after':rng.bit_generator.state})
    np.savez_compressed(dest/f'seed_{seed}_parameters.npz',**parameters)
    np.savez_compressed(dest/f'seed_{seed}_predictions.npz',**predictions)
    write_json(dest/f'seed_{seed}_rng.json',{'initial':initial,'blocks':states})
    write_json(dest/f'seed_{seed}.json',records)
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args();p=json.loads((ROOT/'scaling_settings.json').read_text(encoding='utf-8-sig'))
    checks=self_test(p)
    if args.self_test:print(json.dumps(checks))
    if not args.run:return
    assert p['seeds']==list(range(10101,10113))
    frozen=lock();manifest=OUT/'run_manifest.json';assert not manifest.exists(),'Refuse outcome-dependent restart'
    start=time.time();specs=catalogue(p)
    cat=[{k:v for k,v in s.items() if not isinstance(v,np.ndarray)} for s in specs]
    write_json(OUT/'readout_catalogue.json',cat)
    np.savez_compressed(OUT/'readout_tables.npz',**{s['id']+'_'+key:s[key] for s in specs for key in ['truth','fourier','derivative_table']})
    write_json(manifest,{'status':'started','lock':frozen,'self_tests':checks,'numpy':np.__version__,'python':platform.python_version(),'device':'CPU','dtype':'float64'})
    rows=[]
    for seed in p['seeds']:rows.extend(run_seed(seed,p,specs))
    write_json(OUT/'raw_results.json',rows)
    owned=[x for x in OUT.rglob('*') if x.is_file() and x!=manifest]
    write_json(manifest,{'status':'complete','lock':frozen,'self_tests':checks,'numpy':np.__version__,'python':platform.python_version(),
        'device':'CPU','dtype':'float64','seconds':time.time()-start,'records':len(rows),'readout_conditions':len(specs),
        'output_sha256':{x.relative_to(OUT).as_posix():digest(x) for x in owned}})
    print('Frame-scaling experiment complete',flush=True)


if __name__=='__main__':main()
