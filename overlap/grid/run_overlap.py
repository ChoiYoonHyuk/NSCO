"""Paired fixed-target Gaussian parity study; synthetic observations are streamed."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
try:
    import resource
except ImportError:
    resource = None
from pathlib import Path
import argparse,hashlib,json,math,platform,time
import numpy as np
from scipy.special import erf
import filters
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"known"/"scripts"))
from first_sign import quadrature,scales
filters.erf=erf
ROOT=Path(__file__).resolve().parent
METHODS=['certificate','adjacent_pair','complete_sign','sign_magnitude']

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');tmp.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rng(seed,stream):return np.random.default_rng(np.random.SeedSequence([seed,stream]))
def sign(x):return np.where(x>=0,1.,-1.)
def polar(a):
    u,_,v=np.linalg.svd(a,full_matrices=False);return u@v

def target(base,rho):
    P=np.vstack([np.zeros((1,base.shape[1])),base.copy()])
    P[:,0]*=math.sqrt(1-rho*rho);P[0,0]=rho
    return P

def draw(gen,n,k,D,d):
    q=gen.standard_normal((n,d));q/=np.linalg.norm(q,axis=1,keepdims=True)
    return q,gen.standard_normal((n,k,2,D))

def path(q,W,P):
    K=W@P;state=q;ii=[];vv=[];rows=np.arange(len(q))
    for h in range(W.shape[1]):
        ix=np.einsum('bnd,bd->bn',K[:,h],state).argmax(1)
        ii.append(ix);vv.append(sign(W[rows,h,ix,0]));state=K[rows,h,ix]
    return np.stack(ii,1),np.stack(vv,1)

def gates(W,k,coef):
    g=W[:,1:k,:,0];b=sign(g).mean(2)
    eta=[x['projected_estimate'] for x in coef['coefficients']]
    psi=(b==0)*((np.abs(g).sum(2)>=2)-coef['magnitude']['tail_probability'])/coef['magnitude']['normalizer']
    return [b.prod(1),filters.gate(b,eta,0,False),filters.gate(b,eta,k-2,True),
            filters.gate(b,eta,k-2,True,psi,coef['magnitude']['estimate'])]

def recover(M,rho,d):
    x,w=quadrature(d);_,hp,ho=scales(rho,x,w,d)
    # Known r=(rho,0,...): private Gram is diag(1-rho^2,1,...).
    # E[M]=positive_scalar * Q diag(sqrt(1-rho^2),1,...) H_r.
    # Divide the known right factors and apply polar only on active columns.
    h=np.array([hp]+[ho]*(d-1));root=np.array([math.sqrt(1-rho*rho)]+[1.]*(d-1))
    active=root>1e-12;Q=polar(M[:,active]/(h[active]*root[active])[None])
    P=np.zeros((M.shape[0]+1,d));P[0,0]=rho;P[1:,active]=Q*root[active][None]
    return P

def coefficient_study(p,out):
    file=out/'coefficients.json'
    if file.exists():return json.loads(file.read_text())
    n=p['coefficient_examples'];gap=max(p['depths'])-2
    pp={'coefficient_examples':n,'batch_size':4096,'maximum_gap':gap,
        'coefficient_seed':p['coefficient_seed'],
        'normalized_hoeffding_halfwidth':str(math.sqrt(math.log(2*len(p['dimensions'])*len(p['overlaps'])*(gap+2)/.01)/(2*n)))}
    result=[]
    for d in p['dimensions']:
        for rho in p['overlaps']:
            tic=time.monotonic();row=filters.coefficients(pp,d,rho,len(result))
            row['seconds']=time.monotonic()-tic;row['monte_carlo_examples']=n
            result.append(row)
    write(file,result);return result

def evaluate_public(q,W,k,coef,truth):
    # A public scalar baseline sees candidate values only, and has no trained relation.
    g=W[:,:k,:,0];b=sign(g).mean(2)
    psi=(b==0)*((np.abs(g).sum(2)>=2)-coef['magnitude']['tail_probability'])/coef['magnitude']['normalizer']
    eta=[v['projected_estimate'] for v in coef['coefficients']]
    pred=sign(filters.gate(b,eta,k-2,True,psi,coef['magnitude']['estimate']))
    return float(np.mean(pred!=truth))

def run_seed(p,seed,d,coefficients,out):
    prefix=out/f'd{d}_seed{seed}';resultpath=prefix.with_suffix('.json')
    if resultpath.exists():raise RuntimeError('Refuse overwrite completed observations')
    D=p['D'];depths=p['depths'];kmax=max(depths);rr=rng(seed,101+d)
    base=polar(rr.standard_normal((D-1,d)));targets=[target(base,r) for r in p['overlaps']]
    qt,Wt=draw(rng(seed,201+d),p['test_examples'],kmax,D,d)
    truth=[path(qt,Wt,P) for P in targets]
    cs=[next(c for c in coefficients if c['d']==d and c['rho']==r) for r in p['overlaps']]
    public={(j,k):evaluate_public(qt,Wt,k,cs[j],truth[j][1][:,:k].prod(1)) for j in range(len(cs)) for k in depths}
    shape=(len(cs),len(depths),len(METHODS));M=np.zeros(shape+(D-1,d));gatecost=np.zeros(shape);traincost=np.zeros(shape)
    gate_sq=np.zeros(shape);gate_signal=np.zeros(shape);records=[];frames={};seen=0;train=rng(seed,301+d)
    dataseconds=0.;truthseconds=0.;started=time.monotonic()
    for n in p['budgets']:
        while seen<n:
            size=min(p['batch_size'],n-seen);tic=time.monotonic();q,W=draw(train,size,kmax,D,d)
            noise=np.where(train.random(size)<(1+p['lambda'])/2,1.,-1.)
            delta=(W[:,0,0]-W[:,0,1])/math.sqrt(2);R=delta[:,0,None]*delta[:,1:]
            dataseconds+=time.monotonic()-tic
            for j,P in enumerate(targets):
                tic=time.monotonic();_,v=path(q,W,P);truthseconds+=time.monotonic()-tic
                for h,k in enumerate(depths):
                    Y=v[:,:k].prod(1)*noise
                    # Gate construction timings measured individually, with shared public preparation charged equally.
                    tic=time.monotonic();gg=W[:,1:k,:,0];b=sign(gg).mean(2)
                    eta=[x['projected_estimate'] for x in cs[j]['coefficients']]
                    prep=time.monotonic()-tic
                    for m,name in enumerate(METHODS):
                        tic=time.monotonic()
                        if m==0:f=b.prod(1)
                        elif m==1:f=filters.gate(b,eta,0,False)
                        elif m==2:f=filters.gate(b,eta,k-2,True)
                        else:
                            psi=(b==0)*((np.abs(gg).sum(2)>=2)-cs[j]['magnitude']['tail_probability'])/cs[j]['magnitude']['normalizer']
                            f=filters.gate(b,eta,k-2,True,psi,cs[j]['magnitude']['estimate'])
                        gatecost[j,h,m]+=time.monotonic()-tic+prep
                        tic=time.monotonic();M[j,h,m]+=R.T@((Y*f)[:,None]*q)
                        traincost[j,h,m]+=time.monotonic()-tic
                        gate_sq[j,h,m]+=np.sum(f*f)
                        gate_signal[j,h,m]+=np.sum(f*v[:,1:k].prod(1))
            seen+=size
        for j,rho in enumerate(p['overlaps']):
            for h,k in enumerate(depths):
                trueidx,truev=truth[j];ytrue=truev[:,:k].prod(1)
                for m,name in enumerate(METHODS):
                    tic=time.monotonic();frame=recover(M[j,h,m]/n,rho,d);recoverseconds=time.monotonic()-tic
                    tic=time.monotonic();ii,vv=path(qt,Wt[:,:k],frame);evalseconds=time.monotonic()-tic
                    prediction=vv.prod(1);err=float(np.mean(prediction!=ytrue));ferr=float(np.linalg.norm(frame-targets[j])/math.sqrt(d))
                    row={'seed':seed,'d':d,'D':D,'rho':rho,'k':k,'n':n,'method':name,
                         'final_labels':n,'additional_supervision_labels':0,'clean_error':err,'frame_error_normalized':ferr,
                         'projector_error':float(np.linalg.norm(frame@frame.T-targets[j]@targets[j].T)),
                         'hop_accuracy':(ii==trueidx[:,:k]).mean(0).tolist(),'complete_chain_accuracy':float(np.mean(np.all(ii==trueidx[:,:k],1))),
                         'public_values_only_error':public[j,k],'gate_second_moment':float(gate_sq[j,h,m]/n),
                         'gate_identifying_signal':float(gate_signal[j,h,m]/n),'gate_seconds':float(gatecost[j,h,m]),
                         'moment_seconds':float(traincost[j,h,m]),'recovery_seconds':recoverseconds,'evaluation_seconds':evalseconds,
                         'training_arithmetic_seconds':float(gatecost[j,h,m]+traincost[j,h,m]+recoverseconds),
                         'shared_generation_seconds':dataseconds,'shared_teacher_seconds':truthseconds,
                         'public_coefficient_seconds':cs[j]['seconds'],'public_mc_examples':cs[j]['monte_carlo_examples'],
                         'process_peak_rss_mb':(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 if resource else None),
                         'test_examples':len(qt),'prediction_threshold_met':err<=p['prediction_threshold'],
                         'frame_threshold_met':ferr<=p['frame_threshold']}
                    records.append(row);frames[f'r{rho}_k{k}_m{m}_n{n}']=frame
        write(prefix.with_name(prefix.name+'_progress.json'),{'n':n,'seconds':time.monotonic()-started,'rows':len(records)})
        print(json.dumps({'seed':seed,'d':d,'n':n,'seconds':round(time.monotonic()-started,2)}),flush=True)
    write(resultpath,records);np.savez_compressed(prefix.with_name(prefix.name+'_frames.npz'),**frames)
    return records

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','confirmation']);args=ap.parse_args()
    p=json.loads((ROOT/'protocol.json').read_text());out=ROOT/args.stage;out.mkdir(exist_ok=True)
    if args.stage=='smoke':
        p={**p,'seeds':[99071],'dimensions':[2,4],'overlaps':[0.,.5,1.],
           'budgets':[4096,16384],'coefficient_examples':16384,'test_examples':2048}
    sources=['run_overlap.py','protocol.json','filters.py','../known/scripts/first_sign.py']
    lock={x:sha(ROOT/x) for x in sources};write(out/'execution_lock.json',lock);write(out/'registered_protocol.json',p)
    started=time.monotonic();coefs=coefficient_study(p,out);rows=[]
    for d in p['dimensions']:
        for seed in p['seeds']:rows+=run_seed(p,seed,d,coefs,out)
    write(out/'all_rows.json',rows)
    assert all(sha(ROOT/x)==s for x,s in lock.items())
    write(out/'completion.json',{'status':'complete','stage':args.stage,'rows':len(rows),'seconds':time.monotonic()-started,
       'source_sha256':lock,'numpy':np.__version__,'python':platform.python_version(),'hostname':platform.node(),
       'peak_rss_mb':(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 if resource else None),'threads':2,
       'results_sha256':sha(out/'all_rows.json'),'raw_observations_stored':False})

if __name__=='__main__':main()
