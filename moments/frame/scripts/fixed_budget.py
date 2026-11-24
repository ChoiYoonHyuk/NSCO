"""Fixed fresh first-value/final moments and terminal subspace-orientation study."""
from __future__ import annotations
import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(variable,"2")
import numpy as np
from gaussian_model import (ROOT, centered_R, derivative_first, digest, draw,
    generator, hard_path, kappa_first, make_frame, noise, polar, readout,
    self_test, signs, soft_path, write_json)

OUT = ROOT / "results/fixed_budget"


def test_data(seed,p):
    target=make_frame(generator(seed,p["streams"]["frame"]),p["D"],p["d"])
    q,g,z=draw(generator(seed,p["streams"]["clean_test"]),p["moments"]["test_examples_per_seed"],9,p["D"],p["d"])
    idx,val=hard_path(q,g,z,target)
    path=OUT/f"seed_{seed}_test.npz"
    arrays=dict(q=q,g=g,z=z,target_frame=target,target_indices=idx.astype(np.uint8),target_values=val.astype(np.int8))
    if path.exists():
        with np.load(path) as old:
            assert all(np.array_equal(old[k],v) for k,v in arrays.items())
    else:
        temporary=path.with_name(path.stem+".partial.npz")
        np.savez_compressed(temporary,**arrays)
        temporary.replace(path)
    return target,q,g,z,idx,val


def assess(frame,q,g,z,target_indices,target_values,name,k,theta):
    idx,val=hard_path(q,g[:,:k],z[:,:k],frame)
    raw=soft_path(q,g[:,:k],z[:,:k],frame,theta)
    clean=readout(target_values[:,:k],name)
    hard=readout(val,name)
    soft=readout(signs(raw),name)
    path=(idx!=target_indices[:,:k]).any(1)
    assert np.all((hard==clean)|path)
    disagreement=hard!=soft
    assert np.all(~disagreement|(signs(raw)!=val).any(1))
    metrics={"hard_error":float((hard!=clean).mean()),"soft_error":float((soft!=clean).mean()),
             "hard_soft_disagreement":float(disagreement.mean()),"path_disagreement":float(path.mean())}
    return metrics,idx.astype(np.uint8),raw


def run_first(seed,p):
    out=OUT/"moments";out.mkdir(exist_ok=True)
    assert not (out/f"seed_{seed}.json").exists(),"Refuse to overwrite a seed"
    target,qt,gt,zt,it,vt=test_data(seed,p)
    cfg=p["moments"];sizes=cfg["sample_sizes"];depths=cfg["depths"]
    rng=generator(seed,p["streams"]["moment_training"])
    initial_state=rng.bit_generator.state
    names=[(name,k) for name in cfg["readouts"] for k in depths]
    sums={(name,k):np.zeros_like(target) for name,k in names}
    aux=np.zeros_like(target);counts={key:0 for key in names}
    records,arrays,states=[],{"target_frame":target},[]
    predictions={};seen=0;next_size=0
    s=p["lambda"]*math.sqrt(2/math.pi)/math.sqrt(math.pi)/p["d"]
    for start in range(0,max(sizes),cfg["generation_chunk_size"]):
        state_before=rng.bit_generator.state
        batch=min(cfg["generation_chunk_size"],max(sizes)-start)
        q,g,z=draw(rng,batch,9,p["D"],p["d"])
        idx,val=hard_path(q,g,z,target)
        zeta=noise(rng,batch,p["lambda"]);xi=noise(rng,batch,p["mu"])
        R=centered_R(g[:,0],z[:,0]);a=xi*val[:,0]
        ys={(name,k):zeta*readout(val[:,:k],name) for name,k in names}
        ds={(name,k):derivative_first(g,name,k) for name,k in names}
        if start==0:
            np.savez_compressed(out/f"seed_{seed}_raw_fixture.npz",q=q[:64],g=g[:64],z=z[:64],
                target_indices=idx[:64],target_values=val[:64],zeta=zeta[:64],xi=xi[:64],R=R[:64],
                **{f"Y_{name}_{k}":v[:64] for (name,k),v in ys.items()},
                **{f"derivative_{name}_{k}":v[:64] for (name,k),v in ds.items()})
        cursor=0
        while cursor<batch:
            stop=min(batch,cursor+sizes[next_size]-seen);sl=slice(cursor,stop)
            aux+=R[sl].T@(a[sl,None]*q[sl])
            for key in names:
                sums[key]+=R[sl].T@((ys[key][sl]*ds[key][sl])[:,None]*q[sl])
                counts[key]+=int(np.count_nonzero(ds[key][sl]))
            seen+=stop-cursor;cursor=stop
            if seen!=sizes[next_size]:continue
            n=seen
            aux_frame,aux_rank=polar(aux/n)
            arrays[f"n{n}_first_value_moment"]=aux/n
            arrays[f"n{n}_first_value_frame"]=aux_frame
            arrays[f"n{n}_first_value_singular"]=np.linalg.svd(aux/n,compute_uv=False)
            for name,k in names:
                for method in cfg["methods"]:
                    kap=kappa_first(name,k) if method=="final_derivative" else 1.
                    sig=s*kap if method=="final_derivative" else p["mu"]*math.sqrt(2/math.pi)/math.sqrt(math.pi)/p["d"]
                    moment=sums[name,k]/n if method=="final_derivative" else aux/n
                    frame,rank=polar(moment) if method=="final_derivative" else (aux_frame,aux_rank)
                    row={"seed":seed,"n":n,"depth":k,"readout":name,"method":method,"signal":sig,"kappa":kap,
                         "moment_mse":float(np.linalg.norm(moment/sig-target)**2/p["d"]),
                         "exact_moment_mse":(p["D"]-1)*kap/(n*p["d"]*sig*sig)-1/n,
                         "frame_error":float(np.linalg.norm(frame-target)/math.sqrt(p["d"])),"moment_rank":rank,
                         "nonzero_derivatives":counts[name,k] if method=="final_derivative" else n}
                    metrics,pi,raw=assess(frame,qt,gt,zt,it,vt,name,k,cfg["soft_inverse_temperature"])
                    row.update(metrics);records.append(row)
                    key=f"n{n}_{name}_k{k}_{method}"
                    arrays[key+"_moment"]=moment;arrays[key+"_frame"]=frame
                    arrays[key+"_singular"]=np.linalg.svd(moment,compute_uv=False)
                    predictions[key+"_hard_indices"]=pi;predictions[key+"_soft_raw_values"]=raw
            next_size+=1
            print(json.dumps({"stage":"first_moments","seed":seed,"n":n}),flush=True)
            if next_size==len(sizes):break
        states.append({"start":start,"count":batch,"rng_before":state_before,"rng_after":rng.bit_generator.state})
    np.savez_compressed(out/f"seed_{seed}_parameters.npz",**arrays)
    np.savez_compressed(out/f"seed_{seed}_predictions.npz",**predictions)
    write_json(out/f"seed_{seed}_rng.json",{"initial":initial_state,"blocks":states})
    write_json(out/f"seed_{seed}.json",records)
    write_json(out/f"seed_{seed}_terminal_null.json",[
        {"seed":seed,"depth":k,"n":n,"readout":"terminal","kappa1":0.,
         "first_derivative_identically_zero":True,"unnormalized_moment_norm":0.,"frame_estimate":None}
        for k in depths for n in sizes])
    return records


def main():
    import statistics
    p = json.loads((ROOT / 'fixed_budget_settings.json').read_text(encoding='utf-8-sig'))
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in p['seeds']:
        cached = OUT / 'moments' / f'seed_{seed}.json'
        if cached.exists():
            records.extend(json.loads(cached.read_text()))
        else:
            records.extend(run_first(seed, p))
    print("Readout,depth,final_error,first_sign_error,ratio")
    for readout in p['moments']['readouts']:
        for depth in p['moments']['depths']:
            rates = {}
            for method in p['moments']['methods']:
                values = [r['frame_error'] for r in records if r['readout'] == readout and r['depth'] == depth and r['n'] == 2**18 and r['method'] == method]
                if len(values) != len(p['seeds']):
                    raise ValueError('Incomplete fixed-budget study')
                rates[method] = statistics.mean(values)
            final, first = rates['final_derivative'], rates['first_value']
            print(f"{readout},{depth},{final:.3f},{first:.3f},{final/first:.2f}")

if __name__ == '__main__':
    main()
