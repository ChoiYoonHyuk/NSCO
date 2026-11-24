"""Server-only split construction and frozen text features for the MuSiQue bridge."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import os
import hashlib
import json
import re
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("MUSIQUE_SOURCE", ROOT.parent / "retrieval" / "data"))
BGE = Path(os.environ.get("MUSIQUE_BGE", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"))
SALT = 'controlled-bridge-20261121-v1'
CAPS = {'test': {2:512, 3:256}, 'validation': {2:192,3:128}, 'train': {2:1536,3:768}}
STOP = set('a an the at in of for from and by as it with was is on to which what who this that do did does his her he she had have are were be its has their where how when they or'.split())

def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()

def sha(path):
    obj = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''):
            obj.update(chunk)
    return obj.hexdigest()

def save_json(name, value):
    (ROOT/name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())

@lru_cache(maxsize=150000)
def text_key(text):
    return digest(norm(text))

def tokens(text):
    return set(re.findall(r'[a-z0-9]+', norm(text))) - STOP

def load_rows(part):
    return [json.loads(line) for line in (SOURCE/f'musique_ans_v1.0_{part}.jsonl').open()]

def partition():
    raw = {p:load_rows(p) for p in ['train','dev']}
    records = {}
    for source, rows in raw.items():
        eligible = []
        for i,r in enumerate(rows):
            d = r['question_decomposition']
            k = len(d)
            if k not in (2,3) or any(set(re.findall(r'#(\d+)',x['question'])) != ({str(h)} if h else set()) for h,x in enumerate(d)):
                continue
            paras = sorted(r['paragraphs'],key=lambda p:p['idx'])
            assert [p['idx'] for p in paras] == list(range(len(paras)))
            ordered = [x['paragraph_support_idx'] for x in d]
            texts = {text_key(p['paragraph_text']) for p in paras}
            supports = {text_key(paras[j]['paragraph_text']) for j in ordered}
            if len(supports) != k:
                continue
            q = norm(r['question'])
            comp = {'full:'+q} | {'id:'+str(x['id']) for x in d} | {'sub:'+norm(x['question']) for x in d}
            eligible.append((i,k,q,comp,supports,texts))
        records[source] = sorted(eligible, key=lambda x:digest(SALT+rows[x[0]]['id']))
    used_q, held_comp, held_support, held_candidates = set(),set(),set(),set()
    receipt = {'eligible': {p:dict(Counter(x[1] for x in entries)) for p,entries in records.items()},'splits':{}}
    manifest = {}
    for part, source in [('test','dev'),('validation','train'),('train','train')]:
        chosen = []
        counts = Counter()
        for entry in records[source]:
            i,k,q,comp,supports,texts = entry
            if counts[k] >= CAPS[part][k] or q in used_q or comp & held_comp or texts & held_support or supports & held_candidates:
                continue
            chosen.append(entry)
            used_q.add(q)
            counts[k] += 1
        assert all(counts[k] >= min(96,CAPS[part][k]) for k in (2,3)), (part,counts)
        parts = {'component_overlap_with_preceding':0,'candidate_to_preceding_support_overlap':0,'support_to_preceding_candidate_overlap':0}
        for col, previous,key in [(3,held_comp,'component_overlap_with_preceding'),(5,held_support,'candidate_to_preceding_support_overlap'),(4,held_candidates,'support_to_preceding_candidate_overlap')]:
            parts[key] = len(set().union(*(r[col] for r in chosen)) & previous)
            assert parts[key] == 0
        held_comp.update(set().union(*(r[3] for r in chosen)))
        held_support.update(set().union(*(r[4] for r in chosen)))
        held_candidates.update(set().union(*(r[5] for r in chosen)))
        manifest[part] = {'source':source, 'indices':[r[0] for r in chosen]}
        receipt['splits'][part] = {'counts':dict(counts),'unique_questions':len(chosen), **parts}
    save_json('split.private.json',manifest)
    receipt['split_sha256'] = sha(ROOT/'split.private.json')
    receipt['source_sha256'] = {p:sha(SOURCE/f'musique_ans_v1.0_{p}.jsonl') for p in raw}
    receipt['e5_revision'] = json.loads((SOURCE/'e5/encoder_manifest.json').read_text())['revision']
    receipt['bge_revision'] = BGE.name
    save_json('data_receipt.json',receipt)
    print(json.dumps(receipt), flush=True)
    return raw,manifest

def feature_rows(raw, manifest):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(BGE,local_files_only=True)
    rerank = AutoModelForSequenceClassification.from_pretrained(BGE,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval()
    emb = np.load(SOURCE/'e5/features.npy',mmap_mode='r')
    for part, spec in manifest.items():
        src = spec['source']
        idx = np.load(SOURCE/f'e5/indices_{src}.npz')
        ctx_idx = np.load(SOURCE/f'contexts/musique/indices_{src}.npz')
        ctx = np.load(SOURCE/f'contexts/musique/context_{src}.npy',mmap_mode='r')
        rowmap = {int(v):i for i,v in enumerate(ctx_idx['source_indices'])}
        for k in [2,3]:
            ids = [i for i in spec['indices'] if len(raw[src][i]['question_decomposition']) == k]
            fs, es, ys, ns, map_hash = [],[],[],[],hashlib.sha256()
            for offset in range(0,len(ids),8):
                subset = ids[offset:offset+8]
                pairs=[]
                for i in subset:
                    r=raw[src][i]
                    for p in sorted(r['paragraphs'],key=lambda p:p['idx']):
                        pairs.append([r['question'],p['title']+'\n'+p['paragraph_text']])
                logits=[]
                with torch.inference_mode():
                    for start in range(0,len(pairs),32):
                        batch=tokenizer(pairs[start:start+32],truncation=True,max_length=256,padding=True,return_tensors='pt').to('cuda')
                        logits.extend(rerank(**batch).logits.float().flatten().cpu().tolist())
                li=0
                for i in subset:
                    r=raw[src][i]
                    pp=sorted(r['paragraphs'],key=lambda p:p['idx'])
                    n=len(pp)
                    assert idx['ids'][i]==r['id'] and n==idx['candidate_count'][i]
                    gold=np.array([v['paragraph_support_idx'] for v in r['question_decomposition']])
                    assert np.array_equal(gold,idx['labels'][i,:k])
                    q=emb[idx['q'][i]]
                    p=emb[idx['p'][i,:n]]
                    if k==2:
                        j=rowmap[i]
                        assert ctx_idx['ids'][j]==r['id'] and np.array_equal(ctx_idx['p'][j,:n],idx['p'][i,:n])
                        parent=ctx[j,:n]
                    else:
                        parent=p
                    z=parent+q
                    z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-8)
                    e5=20*(p@q)
                    prefix=20*(z@p.T)
                    score=np.array(logits[li:li+n]);li+=n
                    query=tokens(r['question'])
                    title=[tokens(x['title']) for x in pp]
                    passage=[tokens(x['paragraph_text']) for x in pp]
                    union=[a|b for a,b in zip(title,passage)]
                    df=Counter(w for doc in union for w in doc)
                    def cover(a,b):
                        den=sum(np.log((n+1)/(1+df[w]))+.1 for w in a)
                        return sum(np.log((n+1)/(1+df[w]))+.1 for w in a&b)/max(den,1e-8)
                    direct=np.stack([e5-e5.mean(),score-score.mean(),[cover(query,x) for x in union],[cover(x,query) for x in title]],axis=-1)
                    links=np.array([[cover(b,a) for b in title] for a in passage])
                    edge=np.concatenate([np.broadcast_to(direct,(n,n,4)),(prefix-prefix.mean(1,keepdims=True))[...,None],links[...,None],links.T[...,None]],axis=-1)
                    # Public permutation is unrelated to labels; there are no source positions in model inputs.
                    rng=np.random.default_rng(int(digest(SALT+'order'+r['id'])[:16],16))
                    perm=rng.permutation(n); gold_perm=np.argsort(perm)[gold]
                    assert np.array_equal(perm[gold_perm],gold)
                    map_hash.update(perm.astype('<i4').tobytes())
                    fs.append(np.pad(direct[perm],((0,20-n),(0,0))))
                    es.append(np.pad(edge[perm][:,perm],((0,20-n),(0,20-n),(0,0))))
                    ys.append(gold_perm);ns.append(n)
                assert li==len(logits)
                if offset % 160 == 0:
                    print('features',part,k,offset,len(ids),flush=True)
            np.savez(ROOT/f'{part}_{k}.private.npz',f=np.array(fs,dtype=np.float32),e=np.array(es,dtype=np.float32),g=np.array(ys),n=np.array(ns))
            print('features done',part,k,len(ids),map_hash.hexdigest(),flush=True)

if __name__=='__main__':
    inputs=partition()
    feature_rows(*inputs)
