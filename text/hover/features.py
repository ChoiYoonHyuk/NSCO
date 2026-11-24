"""Apply the frozen three-hop feature map to a public-only HoVer input bag."""
from collections import Counter
import json
import os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoModelForSequenceClassification,AutoTokenizer
from candidates import ROOT,terms,put,sha

E5="f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
BGE="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def encode(texts,path,cross=False):
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    cls=AutoModelForSequenceClassification if cross else AutoModel
    model=cls.from_pretrained(path,torch_dtype=torch.bfloat16,local_files_only=True).cuda().eval()
    out=[]
    step=32 if cross else 96
    with torch.inference_mode():
        for a in range(0,len(texts),step):
            batch=tok(texts[a:a+step],truncation=True,max_length=256 if cross else 512,
                      padding=True,return_tensors="pt").to("cuda")
            pred=model(**batch)
            if cross:
                row=pred.logits.float().flatten()
            else:
                hidden=pred.last_hidden_state.float()
                mean=(hidden*batch.attention_mask[...,None]).sum(1)/batch.attention_mask.sum(1)[:,None]
                row=torch.nn.functional.normalize(mean)
            out.append(row.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def feature_rows(public,embed,bge):
    F,E=[],[]
    for i,r in enumerate(public):
        q=embed["query: "+r["claim"]]
        bag=r["bag"]
        p=np.stack([embed["passage: "+d["title"]+"\n"+d["text"]] for d in bag])
        ctx=p+q
        ctx/=np.maximum(np.linalg.norm(ctx,axis=1,keepdims=True),1e-8)
        e5=20*(p@q);prefix=20*(ctx@p.T);score=bge[i]
        query=set(terms(r["claim"]))
        title=[set(terms(d["title"])) for d in bag]
        body=[set(terms(d["text"])) for d in bag]
        union=[a|b for a,b in zip(title,body)]
        df=Counter(w for x in union for w in x)
        def cover(a,b):
            def count(lst):return sum(np.log(21/(1+df[w]))+.1 for w in sorted(lst))
            return count(a&b)/max(count(a),1e-8)
        direct=np.stack((e5-e5.mean(),score-score.mean(),
                         [cover(query,v) for v in union],[cover(v,query) for v in title]),axis=-1)
        links=np.array([[cover(t,b) for t in title] for b in body])
        e=np.concatenate([np.broadcast_to(direct,(20,20,4)),(prefix-prefix.mean(1,keepdims=True))[:,:,None],
                          links[:,:,None],links.T[:,:,None]],-1)
        F.append(direct);E.append(e)
    return np.array(F,np.float32),np.array(E,np.float32)


def main():
    torch.set_num_threads(4)
    epath=Path(os.environ["HOVER_E5"]);bpath=Path(os.environ["HOVER_BGE"])
    assert epath.name==E5 and bpath.name==BGE
    p=ROOT/"inputs.private.json";r=json.loads((ROOT/"candidate_receipt.json").read_text())
    assert sha(p)==r["candidate_hash"] and r["candidate_count"]==20
    public=json.loads(p.read_text())
    assert all(set(v)=={"claim","bag"} and all(set(t)=={"title","text"} for t in v["bag"]) for v in public)
    texts=[];pairs=[]
    for v in public:
        texts.append("query: "+v["claim"])
        for d in v["bag"]:
            body=d["title"]+"\n"+d["text"]
            texts.append("passage: "+body);pairs.append([v["claim"],body])
    unique=list(dict.fromkeys(texts))
    em=encode(unique,epath)
    print("encoded",len(unique),flush=True)
    bge=encode(pairs,bpath,True).reshape(len(public),20)
    f,e=feature_rows(public,dict(zip(unique,em)),bge)
    assert np.isfinite(f).all() and np.isfinite(e).all()
    dest=ROOT/"features.private.npz";np.savez_compressed(dest,f=f,e=e)
    put("feature_receipt.json",dict(claims=len(public),pairs=len(pairs),unique_encoder_inputs=len(unique),
        direct_shape=list(f.shape),transition_shape=list(e.shape),feature_sha256=sha(dest),code_sha256=sha(Path(__file__)),
        e5_revision=E5,bge_revision=BGE,device=torch.cuda.get_device_name(),torch=torch.__version__,
        labels_read_by_encoder=False,candidate_receipt_sha256=sha(ROOT/"candidate_receipt.json")))
    print("features complete",flush=True)


if __name__=="__main__":main()
