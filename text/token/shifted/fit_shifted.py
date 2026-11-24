"""Single-pass paired successor-key ablation for the signed hop-3 variant."""
from pathlib import Path
import os,sys,time,json,hashlib,math,csv,subprocess
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from sign_variant import records,eval_records,signs_to_labels
P=json.loads((ROOT/'protocol.json').read_text())
ARMS=[(align,g) for align in P['alignment'] for g in P['readouts']]
A=len(ARMS);WIDTH=P['width'];L=P['sequence_length']
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def writecsv(name,rows):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def inputs_and_labels(data):
    ids,q,truth,signs=data
    # Ground-truth pointers are used here to construct the final labels.
    return (ids,q,signs),signs_to_labels(signs.gather(1,truth)),truth
class PointerAttention(nn.Module):
    def __init__(self,seed):
        super().__init__();gen=torch.Generator().manual_seed(seed)
        Q=.2*torch.randn(4,WIDTH,generator=gen);K=.2*torch.randn(4,WIDTH,generator=gen)
        self.query=nn.Parameter(Q.repeat(A,1,1));self.key=nn.Parameter(K.repeat(A,1,1))
        self.decay=nn.Parameter(torch.zeros(A))
        pos=torch.arange(L);gap=pos[:,None]-pos[None,:]
        mask=(gap<=0)|(pos[None,:]==0);mask[:2,0]=False
        self.register_buffer('mask',mask)
        self.register_buffer('dist',torch.log1p(gap.clamp_min(0)).float())
        self.register_buffer('arm',torch.arange(A))
        self.register_buffer('readout',torch.tensor([P['readouts'].index(g) for _,g in ARMS]))
        self.register_buffer('offset',torch.tensor([int(z=='successor') for z,_ in ARMS]))
    def forward(self,inputs,hard=False):
        ids,q,signs=inputs
        # The only difference between paired arms is this fixed lookup offset.
        j=(torch.arange(L,device=q.device)[None]-self.offset[:,None]).clamp_min(0)
        candidates=ids[:,j].permute(1,0,2)
        match=(self.query@self.key.transpose(-1,-2))/math.sqrt(WIDTH)
        table=match[self.arm[:,None,None,None],ids[None,:,:,None],candidates[:,:,None,:]]
        score=(table-F.softplus(self.decay)[:,None,None,None]*self.dist).masked_fill(self.mask,-1e4)
        arms=self.arm[:,None];b=torch.arange(len(q),device=q.device)[None,:]
        if hard:
            links=score.argmax(-1);cur=q.expand(A,-1);seq=[]
            for _ in range(3):
                cur=links[arms,b,cur];seq.append(cur)
            path=torch.stack(seq,dim=-1);v=signs[b[:,:,None],path]
            out=torch.stack((v[:,:,0],v[:,:,-1],v.prod(-1),v.sum(-1).sign()))
            return out[self.readout,self.arm],path
        prob=score.softmax(-1);v=signs.float()[None,:,:,None]
        first=prob@v;second=prob@first;last=prob@second
        parity=prob@(v*(prob@(v*first)))
        majority=(first+second+last-parity)/2
        values=torch.stack([z.squeeze(-1)[arms,b,q] for z in (first,last,parity,majority)])
        return values[self.readout,self.arm]
def evaluate(model,data):
    out=[]
    with torch.inference_mode():
        for j in range(0,P['test_count'],P['batch_size']):
            inputs,ys,truth=inputs_and_labels([v[j:j+P['batch_size']] for v in data])
            labels=ys.T[model.readout];hard,path=model(inputs,hard=True);soft=model(inputs)
            agrees=path==truth
            out.append(torch.stack(((hard!=labels).float(),((soft>=0)!=(labels>0)).float(),
                                    agrees[:,:,0].float(),agrees[:,:,1].float(),agrees[:,:,2].float(),agrees.all(-1).float()),dim=-1))
    return torch.cat(out,dim=1).mean(1).cpu().numpy()
def cpu_frames_and_verify(model,seed,test,results):
    ids,q,t,signs=[v.cpu().numpy() for v in test]
    Q,K,raw=[v.detach().cpu().numpy() for v in [model.query,model.key,model.decay]]
    np.savez(ROOT/f'fits_{seed}.npz',query=Q,key=K,decay=raw)
    def read(v,g):
        return {'first':v[:,0],'last':v[:,-1],'parity':v.prod(-1),'majority':np.sign(v.sum(-1))}[g]
    dif=[]
    for i,(align,g) in enumerate(ARMS):
        current=q.copy();b=np.arange(len(q));path=[];cand=np.arange(1,L)
        M=Q[i]@K[i].T/math.sqrt(WIDTH)
        for h in range(3):
            w=M[ids[b,current,None],ids[:,cand-int(align=='successor')]]-np.logaddexp(0,raw[i])*np.log1p(np.maximum(0,current[:,None]-cand))
            w[cand>=current[:,None]]=-np.inf
            current=np.where(current<2,0,1+w.argmax(1));path.append(current)
        path=np.stack(path,axis=-1);hit=(path==t)
        hard=np.mean(read(signs[b[:,None],path],g)!=read(signs[b[:,None],t],g))
        expected=[hard,*hit.mean(0),hit.all(1).mean()]
        dif.append(float(np.max(np.abs(np.array(expected)-results[i,[0,2,3,4,5]]))))
    return max(dif)
def main():
    assert os.environ.get('PYTHONHASHSEED')=='0'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(P['memory_allocator_cap_MiB']*1024**2/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False
    lock={f:digest(ROOT/f) for f in ('protocol.json','fit_shifted.py','../sign_variant.py','../native_check.py')}
    up=ROOT.parent/'upstream'
    assert subprocess.check_output(['git','-C',str(up),'rev-parse','HEAD']).decode().strip()=='5735bca9545545774098994e21392b51b285bcc5'
    write('registration.json',dict(source_sha256=lock,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name()))
    begin=time.perf_counter();rows=[];aud=[]
    for seed in P['seeds']:
        test=eval_records(seed);rng=np.random.RandomState(500000+seed)
        model=PointerAttention(seed).cuda()
        assert sum(p.numel() for p in model.parameters())==A*33
        opt=torch.optim.Adam(model.parameters(),lr=P['learning_rate'])
        for seen in range(0,P['budgets'][-1]+1,P['batch_size']):
            if seen in P['budgets']:
                obs=evaluate(model,test)
                for (align,g),v in zip(ARMS,obs):
                    rows.append(dict(seed=seed,alignment=align,readout=g,n=seen,parameters=33,examples=512,
                            hard_error=float(v[0]),soft_error=float(v[1]),hop1=float(v[2]),hop2=float(v[3]),hop3=float(v[4]),
                            full_path_accuracy=float(v[5])))
            if seen<P['budgets'][-1]:
                inputs,ys,_=inputs_and_labels(records(rng,P['batch_size']))
                opt.zero_grad();out=model(inputs);labels=ys.T[model.readout]
                loss=-((1+out*labels)/2).clamp_min(1e-7).log().mean(1).sum()
                loss.backward();opt.step()
        diff=cpu_frames_and_verify(model,seed,test,obs)
        assert diff==0
        aud.append(dict(seed=seed,endpoint_max_numpy_discrepancy=diff,
                        frames_sha256=digest(ROOT/f'fits_{seed}.npz')))
        writecsv('results.csv',rows)
        print('seed',seed,'reserved MiB',round(torch.cuda.max_memory_reserved()/2**20,1),'elapsed',round(time.perf_counter()-begin,2),flush=True)
    assert {f:digest(ROOT/f) for f in lock}==lock
    write('audit.json',dict(rows=len(rows),arms=A,estimates_verified=len(aud)*A,frames=aud,results_sha256=digest(ROOT/'results.csv'),
          max_cuda_allocator_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
          max_cuda_allocator_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,elapsed_seconds=time.perf_counter()-begin,
          epochs=1,source_sha256=lock))
if __name__=='__main__':main()

