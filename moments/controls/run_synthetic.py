"""Isolated, fixed-protocol synthetic experiment. Original project is read-only."""
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[v]='2'
import argparse, hashlib, itertools, json, math, platform, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def rng(seed,name): return np.random.default_rng(np.random.SeedSequence([seed,P['streams'][name]]))
def sign(x):return np.where(x>=0,1.,-1.)
def frame(seed,D,d):
    u,r=np.linalg.qr(rng(seed,'frame').standard_normal((D-1,d)))
    return u*np.where(np.diag(r)>=0,1.,-1.)
def draw(r,n,k,D,d):
    q=r.standard_normal((n,d));q/=np.linalg.norm(q,axis=1,keepdims=True)
    g=r.standard_normal((n,k,2));z=r.standard_normal((n,k,2,D-1))
    return q,g,z
def noise(r,n):return np.where(r.random(n)<(1+P['noise_contrast'])/2,1.,-1.)
def hard(q,g,z,A):
    keys=z@A;query=q;ii=[];vv=[];rows=np.arange(len(q))
    for h in range(g.shape[1]):
        j=np.einsum('bnd,bd->bn',keys[:,h],query).argmax(1)
        ii.append(j);vv.append(sign(g[rows,h,j]));query=keys[rows,h,j]
    return np.stack(ii,1),np.stack(vv,1)
def polar(A):
    u,s,v=np.linalg.svd(A,full_matrices=False)
    assert s[0]>0,'An unidentified zero matrix must not receive arbitrary orientation'
    return u@v
def R(g,z):return (g[:,:,None]*(z-z.mean(1,keepdims=True))).sum(1)
def maj(a,b,c):return (a+b+c-a*b*c)/2
def read(v,name):
    if name=='A':return v[:,0]*maj(v[:,1],v[:,2],v[:,3])
    if name=='B':return v[:,1]*maj(v[:,0],v[:,2],v[:,3])
    return v[:,0] if name=='first' else v[:,-1]
def derivative(g,name):
    b=sign(g).mean(2)
    if name=='A':return maj(b[:,1],b[:,2],b[:,3])
    if name=='B':return b[:,1]*(1-b[:,2]*b[:,3])/2
    return np.ones(len(g)) if name=='first' else np.zeros(len(g))
def evaluate(A,target,test,name):
    q,g,z,ti,tv=test
    if A is None:
        output=sign(sign(g).mean(2)[:,0 if name=='first' else -1])
        return {'clean_error':float(np.mean(output!=read(tv,name))), 'frame_error':None,'projector_error':None,'frame_error_raw':None,'projector_error_raw':None,'hop_accuracy':None,'complete_path_accuracy':None}, {'prediction':output.astype(np.int8)}
    ii,vv=hard(q,g,z,A);pred=read(vv,name);d=target.shape[1]
    ferr=float(np.linalg.norm(A-target));perr=float(np.linalg.norm(A@A.T-target@target.T))
    return {'clean_error':float(np.mean(pred!=read(tv,name))), 'frame_error':ferr/math.sqrt(d),'projector_error':perr/math.sqrt(d),'frame_error_raw':ferr,'projector_error_raw':perr,'hop_accuracy':(ii==ti).mean(0).tolist(),'complete_path_accuracy':float((ii==ti).all(1).mean())}, {'prediction':pred.astype(np.int8),'indices':ii.astype(np.uint8)}
def testdata(seed,c,out):
    A=frame(seed,c['D'],c['d']);q,g,z=draw(rng(seed,'test'),P['test_examples'],c['k'],c['D'],c['d']);i,v=hard(q,g,z,A)
    np.savez_compressed(out/f'{seed}_test.npz',q=q,g=g,z=z,target=A,indices=i,values=v)
    return A,(q,g,z,i,v)
def forced(g,z,A):
    keys=z@A;rows=np.arange(len(g));vs=[]
    for initial in (0,1):
        query=keys[:,0,initial]
        for h in range(1,g.shape[1]):
            j=np.einsum('bnd,bd->bn',keys[:,h],query).argmax(1);query=keys[rows,h,j]
        vs.append(sign(g[rows,-1,j]))
    return vs
def same_beta(seed,c,out):
    tic=time.perf_counter();target,test=testdata(seed,c,out);r=rng(seed,'training');budgets=c['budgets'];M={name:np.zeros_like(target) for name in ['A','B']};sq={'A':0.,'B':0.};seen=0;rows=[];pars={'target':target};preds={};chunk=P['generation_chunk'];hashes=[]
    for n in budgets:
        while seen<n:
            m=min(chunk,n-seen);q,g,z=draw(r,m,c['k'],c['D'],c['d']);idx,vals=hard(q,g,z,target);eta=noise(r,m);rr=R(g[:,0],z[:,0]);digest=hashlib.sha256()
            for a in (q,g,z,eta):digest.update(a.tobytes())
            hashes.append({'start':seen,'n':m,'sha256':digest.hexdigest()})
            for name in M:
                deriv=derivative(g,name);Y=eta*read(vals,name);M[name]+=rr.T@((Y*deriv)[:,None]*q);sq[name]+=float(deriv@deriv)
            seen+=m
        for name in M:
            A=polar(M[name]);metrics,pp=evaluate(A,target,test,name);key=f'{name}_n{n}';pars[key]=A;preds.update({key+'_'+k:v for k,v in pp.items()})
            rows.append({'study':'same_beta','seed':seed,'n':n,'readout':name,'method':'first_position_moment','beta':c['beta'],'kappa1':c['kappa_'+name],'empirical_derivative_mass':sq[name]/n,'seconds_since_seed_start':time.perf_counter()-tic,**metrics})
        print(json.dumps({'study':'same_beta','seed':seed,'n':n}),flush=True)
    np.savez_compressed(out/f'{seed}_parameters.npz',**pars);np.savez_compressed(out/f'{seed}_predictions.npz',**preds);write(out/f'{seed}_training_stream_hashes.json',hashes);write(out/f'{seed}_rows.json',rows);return rows
def position(seed,c,out):
    tic=time.perf_counter();target,test=testdata(seed,c,out);D,d,k=c['D'],c['d'],c['k'];r=rng(seed,'training');chunk=P['generation_chunk'];budgets=c['budgets'];M=np.zeros((D-1,D-1));F=np.zeros_like(target);seen=0;first={};sub={};hashes=[]
    # Evaluate known-readout derivatives from input-only public candidate means.
    qs,gs,zs=draw(rng(seed,'selection'),c['unlabeled_selection_inputs'],k,D,d)
    b=sign(gs).mean(2);masses={}
    for name in ['first','terminal']:
        masses[name]=[]
        for j in range(k):
            plus=b.copy();minus=b.copy();plus[:,j]=1.;minus[:,j]=-1.
            masses[name].append(float(np.mean(((read(plus,name)-read(minus,name))/2)**2)))
    select={name:int(np.argmax(values)) for name,values in masses.items()}
    assert select=={'first':0,'terminal':k-1}
    cuts=sorted(set(budgets+[n//2 for n in budgets]))
    for n in cuts:
        while seen<n:
            m=min(chunk,n-seen);q,g,z=draw(r,m,k,D,d);ii,vv=hard(q,g,z,target);eta=noise(r,m)
            F+=R(g[:,0],z[:,0]).T@((eta*vv[:,0])[:,None]*q)
            M+=R(g[:,-1],z[:,-1]).T@((eta*vv[:,-1])[:,None]*z[:,-2].sum(1))
            dig=hashlib.sha256()
            for a in (q,g,z,eta):dig.update(a.tobytes())
            hashes.append({'start':seen,'n':m,'sha256':dig.hexdigest()});seen+=m
        if n in budgets:first[n]=F/n
        if 2*n in budgets:sub[2*n]=M/n
    rows=[];pars={'target':target};preds={}
    for n in budgets:
        A=polar(first[n]);met,pp=evaluate(A,target,test,'first');key=f'first_n{n}';pars[key]=A;preds.update({key+'_'+a:b for a,b in pp.items()})
        rows.append({'study':'position','seed':seed,'n':n,'readout':'first','method':'position_selected_moment','selected_position_1based':select['first']+1,'position_empirical_masses':masses['first'],'selection_labels':0,'selection_unlabeled_inputs':c['unlabeled_selection_inputs'],'subspace_labels':0,'orientation_labels':0,'first_position_labels':n,**met})
        MM=sub[n];eig,U=np.linalg.eigh((MM+MM.T)/2);U=U[:,-d:];O=np.zeros((d,d));rr=rng(seed,'orientation');count=0;use=0
        while use<n//2:
            m=min(chunk,n//2-use);q,g,z=draw(rr,m,k,D,d);ii,vv=hard(q,g,z,target);Y=noise(rr,m)*vv[:,-1];v0,v1=forced(g,z,U);u=(z[:,0,0]-z[:,0,1])@U;u/=np.linalg.norm(u,axis=1,keepdims=True);O+=u.T@((Y*(v0-v1))[:,None]*q);count+=int(np.count_nonzero(v0-v1));use+=m
        A=U@polar(O);met,pp=evaluate(A,target,test,'terminal');key=f'terminal_n{n}';pars.update({key:A,key+'_subspace_moment':MM,key+'_basis':U,key+'_orientation_moment':O/(n//2)});preds.update({key+'_'+a:b for a,b in pp.items()})
        rows.append({'study':'position','seed':seed,'n':n,'readout':'terminal','method':'position_selected_moment','selected_position_1based':select['terminal']+1,'position_empirical_masses':masses['terminal'],'selection_labels':0,'selection_unlabeled_inputs':c['unlabeled_selection_inputs'],'subspace_labels':n//2,'orientation_labels':n//2,'first_position_labels':0,'orientation_nonzero':count,**met})
        met,pp=evaluate(None,target,test,'terminal');preds.update({f'terminal_null_n{n}_'+a:b for a,b in pp.items()})
        rows.append({'study':'position','seed':seed,'n':n,'readout':'terminal','method':'first_position_null_public_prediction','kappa1':0.,'moment_norm':0.,**met})
        print(json.dumps({'study':'position','seed':seed,'n':n,'terminal_error':rows[-2]['clean_error']}),flush=True)
    np.savez_compressed(out/f'{seed}_parameters.npz',**pars);np.savez_compressed(out/f'{seed}_predictions.npz',**preds);write(out/f'{seed}_training_stream_hashes.json',hashes);write(out/f'{seed}_rows.json',rows);write(out/f'{seed}_timing.json',{'seconds':time.perf_counter()-tic});return rows
def torch_predict(torch,A,q,g,z,theta):
    keys=torch.einsum('bhnm,jmd->jbhnd',z,A);query=q[None].expand(len(A),-1,-1);values=[]
    signed=torch.where(g>=0,1.,-1.)
    for h in range(z.shape[1]):
        score=theta*(keys[:,:,h]*query[:,:,None]).sum(-1);w=torch.softmax(score,-1);values.append((w*signed[None,:,h]).sum(-1));query=(w[:,:,:,None]*keys[:,:,h]).sum(2)
    return torch.stack(values,-1)
def sgd(seed,c,out):
    import torch
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    tic=time.perf_counter();target,test=testdata(seed,c,out);D,d,k=c['D'],c['d'],c['k'];B=c['batch_size'];rd=rng(seed,'initialization');init=rd.standard_normal((D-1,d))*.05
    initial=np.stack([np.zeros_like(target),init,np.zeros_like(target),init]);A=torch.nn.Parameter(torch.tensor(initial));opt=torch.optim.SGD([A],lr=c['initial_lr']);r=rng(seed,'sgd_training');rows=[];pars={'target':target,'initial':initial};preds={};traces=[];names=['first','first','terminal','terminal'];starts=['zero','random_sd_0.05','zero','random_sd_0.05'];maxzero=0.
    for step in range(max(c['budgets'])//B):
        q,g,z=draw(r,B,k,D,d);ii,vv=hard(q,g,z,target);eta=noise(r,B);Y=np.stack([eta*vv[:,0],eta*vv[:,0],eta*vv[:,-1],eta*vv[:,-1]])
        outv=torch_predict(torch,A,torch.tensor(q),torch.tensor(g),torch.tensor(z),c['theta']);output=torch.stack([outv[0,:,0],outv[1,:,0],outv[2,:,-1],outv[3,:,-1]])
        loss=((output-torch.tensor(Y))**2/2).mean(1);opt.zero_grad();loss.sum().backward();maxzero=max(maxzero,float(A.grad[2].abs().max()));assert maxzero==0.
        opt.param_groups[0]['lr']=c['initial_lr']*(1+step/c['decay_offset'])**(-c['decay_power']);opt.step();n=(step+1)*B
        assert torch.isfinite(A).all()
        if n not in c['budgets']:continue
        raw=A.detach().numpy().copy();qtest,gtest,ztest,ti,tv=test
        with torch.no_grad():
            soft=[]
            for st in range(0,len(qtest),512):soft.append(torch_predict(torch,A,torch.tensor(qtest[st:st+512]),torch.tensor(gtest[st:st+512]),torch.tensor(ztest[st:st+512]),c['theta']).numpy())
            soft=np.concatenate(soft,1)
        for j,(name,initialization) in enumerate(zip(names,starts)):
            key=f'{name}_{initialization}_n{n}';pars[key+'_native']=raw[j];sp=soft[j,:,0 if name=='first' else -1];preds[key+'_soft']=sp
            zero=bool(np.max(np.abs(raw[j]))==0)
            deployments=['public_mean_unidentified'] if zero else ['native','polar']
            for deployment in deployments:
                AA=None if zero else (raw[j] if deployment=='native' else polar(raw[j]));met,pp=evaluate(AA,target,test,name);preds.update({key+'_'+deployment+'_'+a:b for a,b in pp.items()})
                tie_error=None
                if zero:
                    tie_i,tie_v=hard(qtest,gtest,ztest,raw[j]);tie_error=float(np.mean(read(tie_v,name)!=read(tv,name)))
                rows.append({'study':'sgd','seed':seed,'n':n,'readout':name,'method':'sgd_'+initialization,'initialization':initialization,'deployment':deployment,'literal_zero_hard_argmax_error':tie_error,'soft_error':float(np.mean(sign(sp)!=read(tv,name))),'soft_excess_risk':float(np.mean((sp-P['noise_contrast']*read(tv,name))**2/2)),'zero_matrix':zero,'terminal_zero_gradient_max':maxzero,'seconds_since_seed_start':time.perf_counter()-tic,**met})
        traces.append({'n':n,'loss':loss.detach().numpy().tolist(),'parameter_norms':np.linalg.norm(raw,axis=(1,2)).tolist()});print(json.dumps({'study':'sgd','seed':seed,'n':n,'seconds':time.perf_counter()-tic}),flush=True)
    np.savez_compressed(out/f'{seed}_parameters.npz',**pars);np.savez_compressed(out/f'{seed}_predictions.npz',**preds);write(out/f'{seed}_rows.json',rows);write(out/f'{seed}_trace.json',traces);return rows
def checks():
    bits=np.array(list(itertools.product([-1.,1.],repeat=4)));result={}
    for name in ['A','B']:
        y=read(bits,name);co=[]
        for mask in range(16):co.append(float(np.mean(y*np.prod(bits[:,[j for j in range(4) if mask>>j&1]],axis=1))))
        beta=sum(a*a*2**(-m.bit_count()) for m,a in enumerate(co));kap=sum(a*a*2**(1-m.bit_count()) for m,a in enumerate(co) if m&1)
        assert beta==13/64 and kap==({'A':13/32,'B':5/32}[name]);result[name]={'beta':beta,'kappa1':kap,'fourier':co}
    r=np.random.default_rng(9);q,g,z=draw(r,128,4,8,2);b=sign(g).mean(2)
    for name in ['A','B']:
        plus=b.copy();minus=b.copy();plus[:,0]+=1e-6;minus[:,0]-=1e-6
        assert np.max(np.abs((read(plus,name)-read(minus,name))/2e-6-derivative(g,name)))<1e-9
    import torch
    torch.set_num_threads(2)
    A=torch.tensor(r.standard_normal((1,7,2))*.2,dtype=torch.float64,requires_grad=True)
    tq,tg,tz=[torch.tensor(x,dtype=torch.float64) for x in [q,g,z]]
    out=torch_predict(torch,A,tq,tg,tz,4.)
    # Independent explicit ambient implementation for a nonorthonormal parameter.
    U=A.detach().numpy()[0];query=q@U.T;manual=[]
    for h in range(4):
        score=4.*(z[:,h]*query[:,None]).sum(-1);w=np.exp(score-score.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
        manual.append((w*sign(g[:,h])).sum(1));query=((w[:,:,None]*z[:,h]).sum(1)@U)@U.T
    err=float(np.max(np.abs(out.detach().numpy()[0]-np.stack(manual,1))));assert err<1e-12
    zero=torch.zeros_like(A,requires_grad=True);zeroout=torch_predict(torch,zero,tq,tg,tz,4.);assert np.array_equal(zeroout.detach().numpy()[0],sign(g).mean(2))
    grad=torch.autograd.grad((zeroout[:,:,-1]**2).sum(),zero)[0];assert bool(torch.all(grad==0))
    result['attention_checks']={'independent_ambient_max_error':err,'zero_output_is_public_sign_mean':True,'terminal_zero_gradient_exact':True}
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','same_beta','position','sgd']);args=ap.parse_args();tic=time.time();out=ROOT/'results'/args.stage;out.mkdir(parents=True,exist_ok=True);lock={n:sha(ROOT/n) for n in ['run_synthetic.py','protocol.json']};lpath=out/'execution_lock.json'
    if lpath.exists():assert json.loads(lpath.read_text())==lock,'Frozen source changed'
    else:write(lpath,lock)
    assert not (out/'manifest.json').exists(),'Refuse completed overwrite'
    write(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'host_platform':platform.platform(),'device':'CPU','threads':2});ck=checks();rows=[]
    if args.stage=='smoke':
        for task,fun in [('same_beta',same_beta),('position',position),('sgd',sgd)]:
            c=dict(P[task]);c['budgets']=[2048];dd=out/task;dd.mkdir(exist_ok=True);rows.extend(fun(P['smoke_seeds'][0],c,dd))
    else:
        fun={'same_beta':same_beta,'position':position,'sgd':sgd}[args.stage];c=P[args.stage]
        for seed in c['seeds']:
            assert not (out/f'{seed}_rows.json').exists(),'Refuse seed overwrite';rows.extend(fun(seed,c,out))
    write(out/'rows.json',rows);files=[f for f in out.rglob('*') if f.is_file()];write(out/'manifest.json',{'status':'complete','stage':args.stage,'source_lock':lock,'checks':ck,'seconds':time.time()-tic,'records':len(rows),'sha256':{str(f.relative_to(out)):sha(f) for f in files}});print('COMPLETE',args.stage,flush=True)
if __name__=='__main__':main()
