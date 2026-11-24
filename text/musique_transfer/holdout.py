"""Audit metadata and run only the frozen public feature extractor on selected questions."""
from pathlib import Path
import collections
import hashlib
import json
import os
import re
import sys
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
BRIDGE=Path(os.getenv("MUSIQUE_BRIDGE",HERE.parent / "bridge"))
OLD=Path(os.getenv("MUSIQUE_EARLY",HERE.parent / "controlled"))
sys.path.insert(0,str(BRIDGE))
import prepare as legacy

def save(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def read(path):
    return json.loads(path.read_text())

def features(row):
    d=row["question_decomposition"]
    ps=sorted(row["paragraphs"],key=lambda p:p["idx"])
    gold=[x["paragraph_support_idx"] for x in d]
    c={("id",str(x["id"])) for x in d}|{("sub",legacy.norm(x["question"])) for x in d}
    return (legacy.norm(row["question"]), c, {legacy.text_key(ps[i]["paragraph_text"]) for i in gold},
            {legacy.text_key(p["paragraph_text"]) for p in ps})

def prepare():
    raw={p:legacy.load_rows(p) for p in ("dev","train")}
    past=read(OLD/"splits.json")
    prev=read(BRIDGE/"split.private.json")
    history=[("train",i) for part in past.values() for i in part["original_indices"]]
    history += [(s["source"],i) for s in prev.values() for i in s["indices"]]
    qset,components,support,candidate,ids=set(),set(),set(),set(),set()
    for src,i in set(history):
        row=raw[src][i]
        q,c,s,t=features(row)
        qset.add(q);components.update(c);support.update(s);candidate.update(t);ids.add(row["id"])
    review={"registration_sha256":legacy.sha(HERE/"registration.json"),
            "old_split_sha256":legacy.sha(OLD/"splits.json"),"bridge_split_sha256":legacy.sha(BRIDGE/"split.private.json"),
            "model_lock_sha256":{draw:legacy.sha(BRIDGE / q / "selection_lock.json") for draw,q in [('eight','.'),('one','one_draw')]},
            "strict_legacy_history":len(set(history)),"history_norm_questions":len(qset),"history_unique_ids":len(ids),"sources":{}}
    tiers=["question","core","full"]
    seen=set(qset)
    manifest={}
    group_indices={}
    for src,rows in raw.items():
        eligible=[]
        for i,r in enumerate(rows):
            d=r["question_decomposition"]
            if len(d)!=3 or any(set(re.findall(r"#(\d+)",v["question"]))!=({str(j)} if j else set()) for j,v in enumerate(d)):
                continue
            e=features(r)
            if len(e[2])==3:
                eligible.append((legacy.digest("reviewer-20261121-3hop"+r["id"]),i,e))
        eligible.sort()
        pools={t:[] for t in tiers}
        counts=collections.Counter()
        meta={}
        for _,i,(q,c,s,t) in eligible:
            r=rows[i]
            if q in seen or r["id"] in ids:
                counts["prior_or_duplicate_question"]+=1
                continue
            seen.add(q)
            pools["question"].append(i)
            flags={"component":bool(c&components),"gold_to_gold":bool(s&support),"candidate_to_support":bool(t&support),"support_to_candidate":bool(s&candidate)}
            meta[i]=flags
            if flags["component"] or flags["gold_to_gold"]:continue
            pools["core"].append(i)
            if flags["candidate_to_support"] or flags["support_to_candidate"]:continue
            pools["full"].append(i)
        chosen={t:ps[:512] for t,ps in pools.items()}
        ix=sorted(set().union(*chosen.values()))
        lookup={idx:j for j,idx in enumerate(ix)}
        manifest[src]={"source":src,"indices":ix}
        group_indices[src]={t:[lookup[idx] for idx in sel] for t,sel in chosen.items()}
        review["sources"][src]={"source_rows":len(rows),"eligible_strict3":len(eligible),"pool":{t:len(v) for t,v in pools.items()},
            "sample":{t:len(v) for t,v in chosen.items()},"tier_exposures":{t:{flag:sum(meta[i][flag] for i in arr) for flag in ("component","gold_to_gold","candidate_to_support","support_to_candidate")} for t,arr in chosen.items()},
            "prior_or_duplicate_question":counts["prior_or_duplicate_question"],"unique_for_featurization":len(ix),
            "overlap_old_controlled_full_questions_or_ids":0}
    save("private.manifest.json",manifest)
    save("private.groups.json",group_indices)
    review['private_manifest_sha256']=legacy.sha(HERE/"private.manifest.json")
    review['private_groups_sha256']=legacy.sha(HERE/"private.groups.json")
    source=legacy.SOURCE
    # Match the ordered IDs of the archived native development evaluation.
    ordered_ids = [row["id"] for row in raw["dev"]]
    encoded_ids = json.dumps(ordered_ids, ensure_ascii=False, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded_ids).hexdigest() == "81d498c3fe727fc76b19d0b410424bc20b7a0d53f3e8dc7e6b7b5e2ac74868b1"
    assert len(raw["train"]) == 19938
    official={}
    with zipfile.ZipFile(source/"musique_official.zip") as a:
        for src in ("train","dev"):
            ids_raw={r["id"] for r in raw[src]}
            positive={r["id"] for line in a.open(f"data/musique_full_v1.0_{src}.jsonl") for r in [json.loads(line)] if r["answerable"]}
            # Answerable IDs are a strict coverage check on the extra archive version.
            assert positive==ids_raw
            official["full_answerable_"+src+"_ids"]=len(positive)
        testrows=[json.loads(x) for x in a.open("data/musique_ans_v1.0_test.jsonl")]
        official["test_questions"]=len(testrows)
        official["test_with_gold_decomposition"]=sum(bool(x.get("question_decomposition")) for x in testrows)
    review["historical_exposure"]={"legacy_native_training_refit":len(raw["train"]),"legacy_native_dev_predictions":len(raw["dev"]),**official}
    save("exposure.json",review)
    print(json.dumps(review),flush=True)
    # Use the exact previous feature builder, including its label-independent candidate permutation.
    legacy.ROOT=HERE
    legacy.feature_rows(raw,manifest)

if __name__=="__main__":prepare()
