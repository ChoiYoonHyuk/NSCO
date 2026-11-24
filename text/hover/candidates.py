"""Build a controlled Hotpot paragraph pool and oracle bags entirely on the server."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import os
import re
import unicodedata
import urllib.request

import numpy as np

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/"protocol.json").read_text())
STOP=set("a an the at in of for from and by as it with was is on to which what who this that do did does his her he she had have are were be its has their where how when they or".split())
def norm(v):return " ".join(unicodedata.normalize("NFKC",v).casefold().split())
def terms(v):return [x for x in re.findall(r"[a-z0-9]+",norm(v)) if x not in STOP]
def h(v):return hashlib.sha256(v.encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,v): (ROOT/name).write_text(json.dumps(v,indent=2,allow_nan=False)+"\n")


def prepare():
    path=ROOT/"hover_dev.private.json"
    if not path.exists():
        source=P["annotated_split"].replace("https://github.com/","https://raw.githubusercontent.com/").replace("/blob/","/")
        urllib.request.urlretrieve(source,path)
    source=Path(os.environ["HOTPOT_DEV"])
    hot=json.loads(source.read_text())
    pool={}
    for r in hot:
        for title,sent in r["context"]:
            text=" ".join(v.strip() for v in sent if v.strip())
            if text and (len(text),text)>(len(pool.get(title,"")),pool.get(title,"")):pool[title]=text
    titles=sorted(pool)
    ids={title:j for j,title in enumerate(titles)}
    weight={};counts=defaultdict(list);length=[]
    for j,t in enumerate(titles):
        words=Counter(terms(t+" "+pool[t]));length.append(sum(words.values()))
        for w,tf in words.items():counts[w].append((j,tf))
    doclen=np.array(length);N=len(titles);base=1.2*(.25+.75*doclen/doclen.mean())
    for w,l in counts.items():
        pair=np.asarray(l);ix,tf=pair[:,0],pair[:,1];df=len(ix)
        idf=np.log1p((N-df+.5)/(df+.5))
        weight[w]=(ix,idf*tf*2.2/(tf+base[ix]))
    entries=json.loads(path.read_text())
    byhop=Counter();eligible=[];ex=[]
    for r in entries:
        support={x[0] for x in r["supporting_facts"]}
        byhop[len(support)]+=1
        if len(support)==3 and int(r["num_hops"])==3 and support<=pool.keys():
            eligible.append((h("hover-transfer-v1-select|"+r["uid"]),r,support))
    chosen=[];seen=set();g=[];group=[];pos=Counter()
    for _,r,support in sorted(eligible,key=lambda x:x[0]):
        q=norm(r["claim"])
        if q in seen:continue
        seen.add(q);score=np.zeros(N)
        for w in sorted(set(terms(q))):
            if w in weight:
                ix,val=weight[w];score[ix]+=val
        order=np.argsort(-score,kind="stable")
        neg=[titles[ix] for ix in order if titles[ix] not in support][:17]
        bag=sorted([*support,*neg],key=lambda t:h("hover-transfer-v1-order|"+r["uid"]+"|"+t))
        assert len(set(bag))==20 and set(bag)&support==support
        gold=sorted(bag.index(t) for t in support);g.append(gold);pos.update(gold)
        group.append(r.get("hpqa_id") or r["uid"])
        assert all(t in pool and pool[t] for t in bag)
        # All gold objects look exactly like all other objects to the encoders.
        x=dict(claim=r["claim"],bag=[dict(title=t,text=pool[t]) for t in bag])
        assert set(x)=={"claim","bag"} and all(set(d)=={"title","text"} for d in x["bag"])
        chosen.append(x)
        ex.append([float(score[ids[t]]) for t in bag])
        if len(chosen)==P["sample_cap"]:break
    assert len(chosen)==P["sample_cap"]
    put("inputs.private.json",chosen)
    np.savez_compressed(ROOT/"labels.private.npz",g=np.array(g),cluster=np.array(group),bm25=np.array(ex))
    receipt=dict(official_dev_total=len(entries),three_distinct=byhop[3],all_three_eligible=len(eligible),
            selected=len(chosen),unique_claim_strings=len(seen),antecedent_clusters=len(set(group)),
            largest_cluster=max(Counter(group).values()),pool_source_qa_examples=len(hot),pool_unique_titles=N,
            candidate_count=20,oracle_supports=3,hard_distractors=17,gold_position_histogram=[pos[i] for i in range(20)],
            selected_with_empty_body=0,predictor_schema=["claim","bag[{title,text}]"],
            exact_title_and_body_from_same_pool=True,ignores_support_sentence_indices=True,
            dev_sha256=sha(path),hotpot_sha256=sha(source),candidate_hash=sha(ROOT/"inputs.private.json"),
            labels_hash=sha(ROOT/"labels.private.npz"),protocol_sha256=sha(ROOT/"protocol.json"),
            frozen_weights_sha256=sha(ROOT/"weights.json"),source_sha256=sha(Path(__file__)))
    put("candidate_receipt.json",receipt)
    print({k:receipt[k] for k in ["all_three_eligible","selected","antecedent_clusters","pool_unique_titles"]})


if __name__=="__main__":prepare()
