"""Frozen sample-split control refinement; ordinary SGD is a matched baseline."""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
import argparse,hashlib,json,math,time,platform
import numpy as np
from gaussian_kernels import qr,hard,configure_torch
from soft_attention import torch_values,numpy_attention
R=Path(__file__).resolve().parents[1]
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial')
 t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rng(seed,stream,k):return np.random.default_rng(np.random.SeedSequence([seed,stream,64,4,k]))
def draw(g,n,k,D=64,d=4):
 q=g.normal(size=(n,d));q/=np.linalg.norm(q,axis=1,keepdims=True)
 return q,g.normal(size=(n,k,2,D))
def polar(a):
 u,s,v=np.linalg.svd(a,full_matrices=False);return u@v
def target(seed,p):return np.vstack((np.zeros((1,p['d'])),qr(rng(seed,p['streams']['target'],0),p['D']-1,p['d'])))
def signs(a):return np.where(a>=0,1.,-1.)
def gvalue(v,name):
 if name=='first_value':return v[...,0]
 if name.startswith('parity'):return v.prod(-1)
 assert name=='majority_product6';z=v.reshape(*v.shape[:-1],2,3)
 return ((z.sum(-1)-z.prod(-1))/2).prod(-1)
def derivative(b,name):
 if name=='first_value':return np.ones(len(b))
 if name.startswith('parity'):return b[:,1:].prod(1)
 return (1-b[:,1]*b[:,2])/2*((b[:,3:6].sum(1)-b[:,3:6].prod(1))/2)
def masses(name,k):
 bits=np.array([[1 if mask>>h&1 else-1 for h in range(k)]for mask in range(2**k)])
 coeff=(gvalue(bits,name)[:,None]*np.prod(np.where(((np.arange(2**k)[None,:,None]>>np.arange(k))&1),bits[:,None,:],1),axis=2)).mean(0)
 kappa=sum(c*c*2.**(-(mask.bit_count()-1))for mask,c in enumerate(coeff)if mask&1)
 # Three-valued independent bag means, weights1/4,1/2,1/4.
 from itertools import product
 b=np.array(list(product([-1.,0.,1.],repeat=k-1)));b=np.column_stack((np.zeros(len(b)),b))
 weights=np.prod(np.where(b[:,1:]==0,.5,.25),axis=1);v=derivative(b,name)
 assert abs((weights*v*v).sum()-kappa)<1e-13
 return float(kappa),float((weights*v**4).sum()),coeff
def soft_values(state,q,W,names,p,device):
 import torch
 a=torch.as_tensor(state,device=device,dtype=torch.float64)
 frames=torch.cat((torch.zeros((len(a),1,p['d']),device=device,dtype=a.dtype),a),1)
 val=torch_values(frames,torch.as_tensor(q,device=device),torch.as_tensor(W,device=device),torch.full((len(a),W.shape[1]),p['theta'],device=device,dtype=a.dtype))
 outs=[]
 for i,name in enumerate(names):
  if name=='first_value':outs.append(val[i,:,0])
  elif name.startswith('parity'):outs.append(val[i].prod(-1))
  else:
   z=val[i].reshape(len(q),2,3);outs.append(((z.sum(-1)-z.prod(-1))/2).prod(-1))
 return torch.stack(outs)
def moment(q,W,Y,names,Q=None):
 b=signs(W[...,0]).mean(2);e=(signs(W[:,0,0,0])-signs(W[:,0,1,0]))/2
 delta=W[:,0,0,1:]-W[:,0,1,1:];d1=np.stack([derivative(b,n)for n in names])
 A=np.einsum('jb,bi,bd->jid',Y*d1*e,delta,q)
 if Q is None:return A,None
 control=signs(np.einsum('bi,rjid,bd->rjb',delta,Q,q))
 B=np.einsum('rjb,bi,bd->rjid',control*d1[None]**2*e[None,None]**2,delta,q)
 return A,B
def candidates(Q,U,A,B,n1,mu,p,names):
 out=[];mats=[];alpha=p['alpha_grid'];lam=p['lambda']
 for j,name in enumerate(names):
  for ir,rate in enumerate(p['rate_grid']):
   q=Q[ir,j];final=polar(U[ir,j]);plain=A[j]/(n1*lam*mu[j]);cv=q+(A[j]-lam*B[ir,j])/(n1*lam*mu[j])
   def add(f,a,m):
    out.append({'readout':name,'family':f,'rate':rate,'alpha':a});mats.append(m)
   add('sgd',1.,final);add('pilot',0.,q)
   for a in alpha:
    add('plain_shrink',a,polar((1-a)*q+a*plain))
    add('cv_shrink',a,polar((1-a)*q+a*cv))
    add('sgd_blend',a,polar((1-a)*q+a*final))
 return out,np.stack(mats)
def predictions(mats,records,p,device):
 import torch
 q,W=records;out=[]
 with torch.no_grad():
  for a in range(0,len(mats),16):
   full=np.concatenate((np.zeros((len(mats[a:a+16]),1,p['d'])),mats[a:a+16]),1)
   chunk=[]
   for b in range(0,len(q),512):
    key=torch.einsum('bhni,mid->mbhnd',torch.tensor(W[b:b+512],device=device),torch.tensor(full,device=device))
    state=torch.tensor(q[b:b+512],device=device)[None].expand(len(full),-1,-1);vals=[]
    ss=torch.tensor(signs(W[b:b+512,...,0]),device=device)
    for h in range(W.shape[1]):
     choice=(key[:,:,h]*state[:,:,None,:]).sum(-1).argmax(-1)
     vals.append(torch.gather(ss[None,:,h].expand(len(full),-1,-1),2,choice[:,:,None])[:,:,0])
     state=torch.gather(key[:,:,h],2,choice[:,:,None,None].expand(-1,-1,1,p['d']))[:,:,0]
    chunk.append(torch.stack(vals,-1).cpu().numpy())
   out.append(np.concatenate(chunk,1))
 return np.concatenate(out,0)
def locked(p,stage,out):
 files=p['source_lock_files']+(['fixed_split_choices.json']if stage=='confirmation'else[])
 lock={n:sha(R/n)for n in files};out.mkdir(parents=True,exist_ok=True);path=out/'execution_lock.json'
 if path.exists():assert read(path)==lock,'Source drift after freeze'
 else:write(path,lock);(out/'registered_protocol.json').write_bytes((R/'sample_split_protocol.json').read_bytes())
 return lock
def train(p,case,stage,seed,device,out):
 import torch
 started=time.monotonic();k=case['depth'];names=case['readouts'];J=len(names);nr=len(p['rate_grid']);D=p['D'];d=p['d'];bs=p['batch_size'];n=case['n'];half=n//2
 path=out/'raw'/f"{case['name']}_seed{seed}";path.parent.mkdir(parents=True,exist_ok=True)
 assert not path.with_suffix('.json').exists(),'Never overwrite completed seeds'
 P=target(seed,p);tr=rng(seed,p['streams']['training'],k)
 U=torch.zeros((nr,J,D-1,d),device=device,dtype=torch.float64,requires_grad=True)
 rate=np.array(p['rate_grid']);mu=np.array([masses(name,k)[0]/(d*math.sqrt(math.pi))for name in names])
 A=np.zeros((J,D-1,d));B=np.zeros((nr,J,D-1,d));prefixA=np.zeros_like(A);Q=None
 hashes=[];labels=[];fixtures={};checkpoints={};sgnames=names*nr
 for t in range(n//bs):
  q,W=draw(tr,bs,k,D,d);z=np.where(tr.random(bs)<(1+p['lambda'])/2,1.,-1.);_,v=hard(q,W,P);Y=np.stack([gvalue(v,name)*z for name in names])
  hashes.append(hashlib.sha256(q.tobytes()+W.tobytes()+z.tobytes()).hexdigest());labels.append(Y.astype(np.int8))
  am,bm=moment(q,W,Y,names,Q)
  if t*bs<half:prefixA+=am
  else:A+=am;B+=bm
  F=soft_values(U.reshape(nr*J,D-1,d),q,W,sgnames,p,device).reshape(nr,J,bs)
  loss=(F-torch.tensor(Y,device=device)[None]).square().mean(2).sum()/2
  grad=torch.autograd.grad(loss,U)[0];assert torch.isfinite(grad).all(),'Nonfinite gradient'
  lr=p['step_base']*rate*(1+t/256)**(-.75)
  if t in [0,half//bs-1,half//bs,n//bs-1]:
   fixtures.update({f'q_{t}':q,f'W_{t}':W,f'Y_{t}':Y,f'U_{t}':U.detach().cpu().numpy().copy(),f'grad_{t}':grad.cpu().numpy()})
  with torch.no_grad():U.sub_(torch.tensor(lr,device=device)[:,None,None,None]*grad)
  if (t+1)*bs==half:
   pilot=U.detach().cpu().numpy().copy();Q=np.stack([np.stack([polar(x)for x in a])for a in pilot]);checkpoints['pilot']=pilot
  if(t+1)*bs in [n//4,3*n//4,n]:checkpoints[f'n{(t+1)*bs}']=U.detach().cpu().numpy().copy()
 Ufinal=U.detach().cpu().numpy().copy();spec,mats=candidates(Q,Ufinal,A,B,n-half,mu,p,names)
 # Development rates/alphas use noisy validation only. Confirmation never selects anew.
 source='validation'if stage=='development'else'test';n_eval=p['validation_n']if stage=='development'else p['test_n']
 eq,eW=draw(rng(seed,p['streams'][source],k),n_eval,k,D,d)
 erng=rng(seed,p['streams'][source]+100,k);_,ev=hard(eq,eW,P);clean=np.stack([gvalue(ev,name)for name in names]);z=np.where(erng.random(n_eval)<(1+p['lambda'])/2,1.,-1.);noisy=clean*z
 if stage=='confirmation':
  sel=read(R/'fixed_split_choices.json');chosen=[]
  for name in names:
   for family in p['families']:
    s=sel['choices'][case['name']][name][family]
    chosen.append(next(i for i,x in enumerate(spec)if x['readout']==name and x['family']==family and x['rate']==s['rate']and x['alpha']==s['alpha']))
  spec=[spec[i]for i in chosen];mats=mats[chosen]
 vals=predictions(mats,(eq,eW),p,device);pred=np.stack([gvalue(vals[i],s['readout'])for i,s in enumerate(spec)]).astype(np.int8)
 rows=[]
 for i,s in enumerate(spec):
  j=names.index(s['readout']);h=pred[i];a=mats[i];truth=clean[j]
  row={**s,'case':case['name'],'depth':k,'seed':seed,'stage':stage,'n':n,
   'allocated_training_labels':n,'labels_used':half if s['family']=='pilot'or s['alpha']==0 else n,
   'hard_error':float(np.mean(h!=truth)),
   'validation_noisy_error':float(np.mean(h!=noisy[j]))if stage=='development'else None,
   'calibrated_risk':float(np.mean((p['lambda']*h-p['lambda']*truth)**2)/2),
   'frame_error':float(np.linalg.norm(a-P[1:])/math.sqrt(d)),
   'subspace_squared':float(np.linalg.norm((np.eye(D-1)-P[1:]@P[1:].T)@a)**2/d)}
  rows.append(row)
 # Native ordinary SGD risk is a separate endpoint, with no refit/calibration.
 native={}
 with torch.no_grad():
  chunks=[]
  for st in range(0,n_eval,512):chunks.append(soft_values(Ufinal.reshape(nr*J,D-1,d),eq[st:st+512],eW[st:st+512],sgnames,p,device).cpu().numpy())
  f=np.concatenate(chunks,1).reshape(nr,J,n_eval)
  native={'risk':(.5*((f-p['lambda']*clean[None])**2).mean(-1)).tolist(),'classification_error':((np.where(f>=0,1,-1)!=clean[None]).mean(-1)).tolist()}
 np.savez_compressed(str(path)+'_states.npz',P=P,Q=Q,Ufinal=Ufinal,A=A,B=B,mu=mu,prefixA=prefixA,mats=mats,**checkpoints)
 np.savez_compressed(str(path)+'_fixtures.npz',**fixtures)
 np.savez_compressed(str(path)+'_evaluation.npz',q=eq,W=eW,clean=clean,noisy=noisy,pred=pred,native_soft=f)
 np.savez_compressed(str(path)+'_labels.npz',Y=np.concatenate(labels,1))
 write(str(path)+'_stream.json',{'batch_sha256':hashes,'rng_final':tr.bit_generator.state})
 write(str(path)+'_meta.json',{'spec':spec,'native':native,'seconds':time.monotonic()-started,'case':case,'seed':seed})
 write(path.with_suffix('.json'),rows)
 print(json.dumps({'stage':stage,'case':case['name'],'seed':seed,'status':'complete','seconds':time.monotonic()-started}),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['confirmation']);ap.add_argument('--device',default='cuda');args=ap.parse_args();p=read(R/'sample_split_protocol.json')
 runtime=configure_torch(args.device);start=time.monotonic()
 stage=args.mode;out=R/f'results/split_{stage}';lock=locked(p,stage,out);assert not(out/'run_manifest.json').exists()
 for seed in p[stage+'_seeds']:
  for case in p['cases']:train(p,case,stage,seed,args.device,out)
 rows=[]
 for seed in p[stage+'_seeds']:
  for case in p['cases']:rows+=read(out/'raw'/f"{case['name']}_seed{seed}.json")
 write(out/'raw_results.json',rows)
 write(out/'run_manifest.json',{'status':'complete','seconds':time.monotonic()-start,'protocol_sha256':sha(R/'sample_split_protocol.json'),'source_lock':lock,
 'runtime':runtime,'output_sha256':{f.relative_to(out).as_posix():sha(f)for f in sorted(out.rglob('*'))if f.is_file()}})
if __name__=='__main__':main()
