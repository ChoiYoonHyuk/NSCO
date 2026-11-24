"""Check every saved endpoint in full ambient coordinates and every recorded selection."""
import csv,json
import numpy as np
from calibrate import ROOT,P,D,K,RHO,NGRID,AGRID,rng,basis,rot,target,full_block,sha,write
O=ROOT/'production'
def explicit(q,W,F):
    # Flatten the candidate bags for projection, then select with explicit inner product comparisons.
    keys=(W.reshape(-1,D)@F).reshape(len(q),K,2,2);idx=[];vals=[];b=np.arange(len(q));s=q
    for j in range(K):
        z=keys[:,j]
        score1=np.sum(s*z[:,1],axis=1);score0=np.sum(s*z[:,0],axis=1)
        take=(score1>score0).astype(int);idx.append(take);s=z[b,take];vals.append(np.sign(W[b,j,take,0]))
    return np.stack(idx,1),np.prod(np.stack(vals,1),axis=1)
def main():
    lock=json.loads((O/'registration.json').read_text())['sha256'];done=json.loads((O/'completion.json').read_text())
    for n,h in lock.items():assert sha(ROOT/n)==h
    for n,h in done['output_sha256'].items():assert sha(O/n)==h
    frame_n=select_n=0;peak=0.
    for seed in P['seeds']:
        B=basis(seed,D);theta=rng(seed,567).uniform(0,360);V=rot(theta)
        qt,Wt,_,_=full_block(seed,859,P['test_count'],B,V)
        outcomes=list(csv.DictReader((O/f'outcomes_{seed}.csv').open()))
        scores=list(csv.DictReader((O/f'selection_{seed}.csv').open()))
        fs=np.load(O/f'fits_{seed}.npz')
        for o in outcomes:
            n=int(o['n']);r=float(o['true_rho']);i=RHO.index(r);a=o['arm'];h=float(o['chosen_rho']);angle=float(o['chosen_degrees'])
            assert int(o['unique_final_labels'])==n==int(o['fit_labels'])+int(o['validation_labels']) and int(o['additional_supervision_labels'])==0
            if a in ['norm_given_direction','joint_grid']:
                ss=[z for z in scores if int(z['n'])==n and float(z['true_rho'])==r and z['arm']==a]
                cand=[(float(z['candidate_norm']),float(z['candidate_angle'])) for z in ss]
                expected=[(v,theta) for v in NGRID] if a=='norm_given_direction' else [(v,t) for v in NGRID for t in ([0.] if v==0 else AGRID)]
                assert cand==expected
                selected=min(ss,key=lambda z:int(z['noisy_validation_errors']))
                assert int(selected['noisy_validation_errors'])==int(o['noisy_validation_errors'])
                assert float(selected['candidate_norm'])==h and float(selected['candidate_angle'])==angle
                select_n+=1
            if n!=max(P['budgets']):continue
            F=fs[f'r{i}_{a}'];truth=target(B[:,:2],r)@V.T
            it,yt=explicit(qt,Wt,truth);ip,yp=explicit(qt,Wt,F);hit=it==ip
            assert (yt!=yp).sum()==int(o['clean_test_errors'])
            assert hit.all(1).sum()==int(o['full_path_correct'])
            target_row=h*np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
            assert np.max(np.abs(F[0]-target_row))<1e-12
            residual=abs(np.linalg.norm(truth-F)/np.sqrt(2)-float(o['normalized_frame_error']))
            peak=max(peak,residual,*[abs(hit[:,j].mean()-float(o['hop'+str(j+1)])) for j in range(K)])
            frame_n+=1
    assert peak<1e-11
    write(ROOT/'frame_audit.json',dict(verified_endpoint_frames=frame_n,verified_grid_selections=select_n,
          exact_integer_errors_and_path_counts=True,max_floating_metric_delta=peak,
          original_source_and_result_hashes_verified=True,refit_counts_all_unique_labels=True))
    print(frame_n,'endpoint frames,',select_n,'candidate grids checked; max difference',peak)
if __name__=='__main__':main()

