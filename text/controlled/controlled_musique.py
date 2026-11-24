"""Frozen-feature MuSiQue controlled-readout diagnostic; all data stays on server.

No natural-language answer task is evaluated. Training sees only declared labels.
Gold metadata is used for splitting, label generation and evaluation.
"""
from pathlib import Path
import argparse, hashlib, json, math, os, platform, re, time, unicodedata
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent / 'retrieval'
PROTOCOL={
 'study':'MuSiQue-derived strict-two-hop controlled-readout diagnostic',
 'version':1,'split_salt':'additional-20261120-text-v1',
 'train_questions':2560,'selection_questions':256,'test_questions':512,
 'seeds':[1101,2202,3303],'readouts':['first','last','parity'],
 'arms':['final','first_value','first_index','all_indices'],
 'epochs':32,'batch_size':128,'rank':64,'learning_rate':0.003,
 'weight_decay':0.0001,'temperature':20.0,'residual_scale':1.0,
 'auxiliary_weight':1.0,'gradient_clip':1.0,
 'training_values':'One independent Rademacher scalar per candidate paragraph, resampled each epoch; same values/order across arms and readouts within seed.',
 'architecture':'Frozen normalized E5 query/paragraph/question-candidate-context embeddings. One shared rank-64 bilinear residual, base scores 20*cosine. First query q; second query normalized(q+context(first)). Exact marginal over all distinct ordered pairs for soft loss; greedy rollout for primary retrieval evaluation.',
 'loss':'Binary cross entropy for final generated label; value arm adds binary cross entropy for selected first scalar; index arms add first CE or mean(first CE, gold-prefix second CE). Fixed weights, no test tuning.',
 'model_selection':'No checkpoint selection. Fixed 32-epoch endpoint. Selection split used only for prespecified diagnostic evaluation after training, never to choose settings.',
 'split':'New question-level partition of official train; purge decomposition IDs/questions and held support paragraphs from training candidates. Selection purged against test. Official dev unused.',
 'inference':'Greedy first index then greedy second conditioned on predicted first. No gold prefix used for primary metrics.',
 'evaluation':'Exact expected clean classification error over independent value bits (0 if selected relevant index/support set is correct, otherwise 1/2); separate sampled 64-bit-assignment check. Public-only baseline integrates uniformly over candidates. No labels used to choose model.',
 'interpretation':'Descriptive controlled diagnostic. Source questions and frozen representations were used by previous experiments. Not official MuSiQue QA; no claim of an untouched benchmark or Gaussian realization.',
 'sources':['https://github.com/stonybrooknlp/musique','https://arxiv.org/abs/2108.00573'],
}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFKC',str(s)).casefold().split())
def write_once(path,data):
 text=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
 if path.exists():assert path.read_text()==text, f'Immutable file differs: {path}'
 else:path.write_text(text)
def read_rows():return [json.loads(s) for s in (SOURCE/'data/musique_ans_v1.0_train.jsonl').open()]
def components(r):
 return {'composed:'+norm(r['question'])}|{'id:'+str(d['id']) for d in r['question_decomposition']}|{'question:'+norm(d['question']) for d in r['question_decomposition']}
def texts(r,support=False):
 targets={d['paragraph_support_idx'] for d in r['question_decomposition']}
 return {digest(norm(p['paragraph_text'])) for p in r['paragraphs'] if not support or p['idx'] in targets}
def union(values,ids):return set().union(*(values[i] for i in ids)) if ids else set()

def prepare():
 rows=read_rows();idx=dict(np.load(SOURCE/'data/contexts/musique/indices_train.npz',allow_pickle=False))
 eligible=[]
 for p,r in enumerate(rows):
  d=r['question_decomposition']
  if len(d)==2 and [set(re.findall(r'#(\d+)',x['question'])) for x in d]==[set(),{'1'}] and len({x['paragraph_support_idx'] for x in d})==2 and len(texts(r,True))==2:eligible.append(p)
 c={i:components(rows[i]) for i in eligible};s={i:texts(rows[i],True) for i in eligible};t={i:texts(rows[i]) for i in eligible}
 order=sorted(eligible,key=lambda i:digest(PROTOCOL['split_salt']+rows[i]['id']))
 test=order[:PROTOCOL['test_questions']]; testset=set(test);tc=union(c,test);ts=union(s,test)
 pool=[i for i in order if i not in testset and not c[i]&tc and not t[i]&ts]
 selection=pool[:PROTOCOL['selection_questions']];held=set(test+selection);hc=union(c,test+selection);hs=union(s,test+selection)
 trainpool=[i for i in pool if i not in held and not c[i]&hc and not t[i]&hs]
 train=trainpool[:PROTOCOL['train_questions']]
 print(json.dumps({'stage':'metadata_only_split_feasibility','eligible':len(eligible),'test':len(test),'selection':len(selection),'pool_after_test':len(pool),'train_pool':len(trainpool)}),flush=True)
 assert len(train)==PROTOCOL['train_questions'] and len(selection)==PROTOCOL['selection_questions']
 assert not union(c,train)&hc and not union(t,train)&hs
 assert not union(c,selection)&tc and not union(t,selection)&ts
 mapping={int(original):pos for pos,original in enumerate(idx['source_indices'])}
 assert all(i in mapping for i in train+selection+test)
 splits={k: {'original_indices':v,'context_indices':[mapping[i] for i in v],'ids':[rows[i]['id'] for i in v]} for k,v in [('train',train),('selection',selection),('test',test)]}
 for name,split in splits.items():
  assert idx['ids'][split['context_indices']].tolist()==split['ids']
  assert idx['labels'][split['context_indices'],:2].tolist()==[[x['paragraph_support_idx'] for x in rows[i]['question_decomposition']] for i in split['original_indices']]
 receipt={'eligible':len(eligible),'pool_after_test_purge':len(pool),'train_pool_after_all_purges':len(trainpool),'counts':{k:len(v['ids']) for k,v in splits.items()},'component_train_held_overlap':0,'candidate_train_held_support_overlap':0,'component_selection_test_overlap':0,'candidate_selection_test_support_overlap':0,'official_dev_read':False,'source_sha256':sha(SOURCE/'data/musique_ans_v1.0_train.jsonl'),'indices_sha256':sha(SOURCE/'data/contexts/musique/indices_train.npz'),'features_sha256':sha(SOURCE/'data/e5/features.npy'),'context_sha256':sha(SOURCE/'data/contexts/musique/context_train.npy'),'encoder_manifest':json.loads((SOURCE/'data/e5/encoder_manifest.json').read_text()),'source_code_sha256':sha(__file__)}
 write_once(ROOT/'protocol.json',PROTOCOL);write_once(ROOT/'splits.json',splits);write_once(ROOT/'data_receipt.json',receipt)
 print(json.dumps(receipt),flush=True)

def load_data(device):
 import torch
 splits=json.loads((ROOT/'splits.json').read_text());idx=dict(np.load(SOURCE/'data/contexts/musique/indices_train.npz',allow_pickle=False));features=np.load(SOURCE/'data/e5/features.npy',mmap_mode='r');contexts=np.load(SOURCE/'data/contexts/musique/context_train.npy',mmap_mode='r')
 out={}
 for name,split in splits.items():
  pos=np.asarray(split['context_indices']);q=np.array(features[idx['q'][pos]]);p=np.array(features[idx['p'][pos]]);c=np.array(contexts[pos]);labels=idx['labels'][pos,:2];counts=idx['candidate_count'][pos]
  out[name]={k:torch.as_tensor(v,device=device) for k,v in dict(q=q,p=p,c=c,labels=labels,counts=counts).items()}
  out[name]['q']=torch.nn.functional.normalize(out[name]['q'],dim=-1);out[name]['p']=torch.nn.functional.normalize(out[name]['p'],dim=-1);out[name]['c']=torch.nn.functional.normalize(out[name]['c'],dim=-1)
 return out

def run():
 import torch
 import torch.nn.functional as F
 torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 assert json.loads((ROOT/'protocol.json').read_text())==PROTOCOL
 lock={'protocol_sha256':sha(ROOT/'protocol.json'),'source_sha256':sha(__file__),'data_receipt_sha256':sha(ROOT/'data_receipt.json'),'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),'device':torch.cuda.get_device_name(0)}
 write_once(ROOT/'execution_lock.json',lock)
 device='cuda';data=load_data(device);dim=data['train']['q'].shape[-1];width=data['train']['p'].shape[1];rank=PROTOCOL['rank'];batch=PROTOCOL['batch_size'];N=len(data['train']['q']);n_test=len(data['test']['q'])
 class Retriever(torch.nn.Module):
  def __init__(self,seed):
   super().__init__();g=torch.Generator(device=device).manual_seed(seed)
   self.A=torch.nn.Parameter(torch.randn(dim,rank,device=device,generator=g)/np.sqrt(dim));self.B=torch.nn.Parameter(torch.zeros(dim,rank,device=device))
  def forward(self,d,ind):
   q,p,c=d['q'][ind],d['p'][ind],d['c'][ind];mask=torch.arange(width,device=device)[None,:]<d['counts'][ind,None]
   second=F.normalize(q[:,None,:]+c,dim=-1)
   # A shared bilinear residual is used at both positions; scalar values never enter routing.
   pB=p@self.B
   s1=PROTOCOL['temperature']*((q[:,None,:]*p).sum(-1)+PROTOCOL['residual_scale']*((q@self.A)[:,None,:]*pB).sum(-1))
   s2=PROTOCOL['temperature']*(torch.bmm(second,p.transpose(1,2))+PROTOCOL['residual_scale']*torch.bmm(second@self.A,pB.transpose(1,2)))
   s1=s1.masked_fill(~mask,-1e4)
   valid=mask[:,None,:]&~torch.eye(width,dtype=torch.bool,device=device)[None,:,:]
   s2=s2.masked_fill(~valid,-1e4)
   return s1,s2,mask
 def readout(p1,p2,values,which):
  a=(p1*values).sum(-1);conditional=torch.bmm(p2,values[:,:,None]).squeeze(-1)
  if which=='first':return a
  if which=='last':return (p1*conditional).sum(-1)
  return (p1*values*conditional).sum(-1)
 def bce(pred,target):return F.binary_cross_entropy(((pred+1)*.5).clamp(1e-6,1-1e-6),(target+1)*.5)
 def eval_model(model,which,split,seed):
  d=data[split];length=len(d['q']);arrays=[];bitg=torch.Generator(device=device).manual_seed(970000+seed)
  torch.cuda.synchronize();t=time.perf_counter()
  with torch.no_grad():
   for start in range(0,length,batch):
    ind=torch.arange(start,min(start+batch,length),device=device);s1,s2,mask=model(d,ind);i=s1.argmax(-1);j=s2[torch.arange(len(i),device=device),i].argmax(-1);gold=d['labels'][ind];correct1=i==gold[:,0];correct2=j==gold[:,1];chain=correct1&correct2;setgood=chain|((i==gold[:,1])&(j==gold[:,0]));relevant={'first':correct1,'last':correct2,'parity':setgood}[which]
    error=.5*(~relevant).float();v=torch.randint(0,2,(len(i),64,width),generator=bitg,device=device).float()*2-1
    row=torch.arange(len(i),device=device)[:,None];bit=torch.arange(64,device=device)[None,:]
    first=v[row,bit,i[:,None]];last=v[row,bit,j[:,None]];gfirst=v[row,bit,gold[:,0,None]];glast=v[row,bit,gold[:,1,None]]
    pred={'first':first,'last':last,'parity':first*last}[which];y={'first':gfirst,'last':glast,'parity':gfirst*glast}[which]
    sampled=(pred!=y).float().mean(-1)
    # Public-only control uses bits and readout, but no text or gold routing.
    sm=(v*mask[:,None,:]).sum(-1);cnt=mask.sum(-1)[:,None]
    pub=sm/cnt if which!='parity' else (sm.square()-cnt)/(cnt*(cnt-1))
    puberr=(torch.where(pub>=0,1.,-1.)!=y).float().mean(-1)
    # Gold-prefix metric is explicitly secondary.
    teacher=s2[torch.arange(len(i),device=device),gold[:,0]].argmax(-1)==gold[:,1]
    exact_public=[]
    for count in d['counts'][ind].cpu().tolist():
     mass=0.
     for plus in range(count+1):
      smv=2*plus-count;mean=smv/count if which!='parity' else (smv*smv-count)/(count*(count-1))
      mass+=math.comb(count,plus)/(2**count)*(.5-.5*abs(mean))
     exact_public.append(mass)
    pub_exact=torch.tensor(exact_public,device=device)
    arrays.append(torch.stack([error,correct1.float(),correct2.float(),chain.float(),setgood.float(),sampled,puberr,teacher.float(),pub_exact,i.float(),j.float()],-1).cpu().numpy())
  torch.cuda.synchronize();elapsed=time.perf_counter()-t
  arr=np.concatenate(arrays);names=['clean_error_exact','hop1_accuracy','hop2_rollout_accuracy','complete_chain_accuracy','all_support_at_2','clean_error_64bit_check','public_only_error_64bit','hop2_gold_prefix_accuracy','public_only_error_exact','selected_first','selected_second']
  return {n:float(arr[:,k].mean()) for k,n in enumerate(names[:9])}|{'inference_and_evaluation_seconds':elapsed},arr,names
 (ROOT/'checkpoints').mkdir(exist_ok=True);(ROOT/'question_metrics').mkdir(exist_ok=True)
 outfile=ROOT/'runs.jsonl'
 done=set()
 if outfile.exists():done={(x['seed'],x['readout'],x['arm']) for x in map(json.loads,outfile.read_text().splitlines())}
 for seed in PROTOCOL['seeds']:
  for which in PROTOCOL['readouts']:
   for arm in ['frozen']+PROTOCOL['arms']:
    if (seed,which,arm) in done:continue
    model=Retriever(seed);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();losses=[];bit_hash=hashlib.sha256()
    if arm!='frozen':
     optimizer=torch.optim.AdamW(model.parameters(),lr=PROTOCOL['learning_rate'],weight_decay=PROTOCOL['weight_decay'])
     gen=torch.Generator(device=device).manual_seed(seed+45000)
     for epoch in range(PROTOCOL['epochs']):
      order=torch.randperm(N,generator=gen,device=device);values=torch.randint(0,2,(N,width),generator=gen,device=device).float()*2-1;total=0.;bit_hash.update(order.cpu().numpy().tobytes());bit_hash.update(values.cpu().numpy().tobytes())
      for begin in range(0,N,batch):
       ind=order[begin:begin+batch];s1,s2,_=model(data['train'],ind);p1=s1.softmax(-1);p2=s2.softmax(-1);v=values[ind];gold=data['train']['labels'][ind];ii=torch.arange(len(ind),device=device);a=v[ii,gold[:,0]];b=v[ii,gold[:,1]];y={'first':a,'last':b,'parity':a*b}[which];prediction=readout(p1,p2,v,which);loss=bce(prediction,y)
       if arm=='first_value':loss=loss+PROTOCOL['auxiliary_weight']*bce((p1*v).sum(-1),a)
       elif arm=='first_index':loss=loss+PROTOCOL['auxiliary_weight']*F.cross_entropy(s1,gold[:,0])
       elif arm=='all_indices':loss=loss+PROTOCOL['auxiliary_weight']*.5*(F.cross_entropy(s1,gold[:,0])+F.cross_entropy(s2[ii,gold[:,0]],gold[:,1]))
       optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),PROTOCOL['gradient_clip']);optimizer.step();total+=float(loss.detach())*len(ind)
      losses.append(total/N)
      if epoch in [0,7,15,31]:print(json.dumps({'seed':seed,'readout':which,'arm':arm,'epoch':epoch+1,'loss':losses[-1],'seconds':time.perf_counter()-start}),flush=True)
    torch.cuda.synchronize();train_seconds=time.perf_counter()-start;tag=f'{seed}_{which}_{arm}';checkpoint=ROOT/'checkpoints'/f'{tag}.pt';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint)
    selection,_,_=eval_model(model,which,'selection',seed);test,arr,names=eval_model(model,which,'test',seed)
    np.savez_compressed(ROOT/'question_metrics'/f'{tag}.npz',metrics=arr,columns=np.asarray(names))
    row={'seed':seed,'readout':which,'arm':arm,'train_unique_questions':0 if arm=='frozen' else N,'train_generated_final_labels':0 if arm=='frozen' else N*PROTOCOL['epochs'],'train_generated_first_value_labels':N*PROTOCOL['epochs'] if arm=='first_value' else 0,'train_unique_support_index_annotations':N*(2 if arm=='all_indices' else 1) if arm in ['first_index','all_indices'] else 0,'updates':0 if arm=='frozen' else (N//batch)*PROTOCOL['epochs'],'epochs':0 if arm=='frozen' else PROTOCOL['epochs'],'selection_questions':len(data['selection']['q']),'test_questions':n_test,'training_bits_and_order_sha256':bit_hash.hexdigest(),'checkpoint_sha256':sha(checkpoint),'losses':losses,'training_seconds':train_seconds,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'selection':selection,'test':test}
    with outfile.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    print('COMPLETE',json.dumps({k:row[k] for k in ['seed','readout','arm','training_seconds','test']}),flush=True)
 write_once(ROOT/'completed.json',{'runs':len(outfile.read_text().splitlines()),'expected_runs':len(PROTOCOL['seeds'])*len(PROTOCOL['readouts'])*(1+len(PROTOCOL['arms'])),'runs_sha256':sha(outfile),'source_sha256':sha(__file__)})

def report():
 rows=[json.loads(x) for x in (ROOT/'runs.jsonl').read_text().splitlines()];assert len(rows)==45
 from scipy.stats import t
 for seed in PROTOCOL['seeds']:assert len({r['training_bits_and_order_sha256'] for r in rows if r['seed']==seed and r['arm']!='frozen'})==1
 summaries=[]
 for which in PROTOCOL['readouts']:
  for arm in ['frozen']+PROTOCOL['arms']:
   cell=[r for r in rows if r['readout']==which and r['arm']==arm];s={'readout':which,'arm':arm,'seeds':[r['seed'] for r in cell]}
   for metric in cell[0]['test']:
    x=np.asarray([r['test'][metric] for r in cell]);s[metric]={'mean':float(x.mean()),'seed_values':x.tolist(),'sd':float(x.std(ddof=1)),'t95_halfwidth':float(t.ppf(.975,2)*x.std(ddof=1)/np.sqrt(3))}
   s['training_seconds']=[r['training_seconds'] for r in cell];summaries.append(s)
 write_once(ROOT/'summary.json',{'status':'complete','runs':45,'interval_scope':'Pointwise Student-t variation across 3 optimization seeds on one fixed question split; not a question-population CI. No confirmatory significance claim.','cells':summaries,'data_receipt':json.loads((ROOT/'data_receipt.json').read_text()),'protocol':PROTOCOL})
 print(json.dumps(summaries),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run','report']);args=a.parse_args();globals()[args.mode]()
