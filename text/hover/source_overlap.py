"""Audit overlap with preceding MuSiQue fit and Hotpot evaluation using aggregate counts."""
import json
import os
from pathlib import Path
import numpy as np
from candidates import ROOT,norm,put,sha


def main():
    inputs=json.loads((ROOT/"inputs.private.json").read_text())
    lab=np.load(ROOT/"labels.private.npz")
    idx=lab["g"]; cluster=lab["cluster"].tolist()
    splitpath=Path(os.environ["MUSIQUE_SPLIT"])
    source=Path(os.environ["MUSIQUE_TRAIN"])
    train=json.loads(splitpath.read_text())["train"]
    assert train["source"]=="train"
    raw=[json.loads(l) for l in source.open()]
    qs=set();titles=set();body=set();sup_titles=set();sup_body=set()
    for i in train["indices"]:
        r=raw[i];qs.add(norm(r["question"]))
        supports={v["paragraph_support_idx"] for v in r["question_decomposition"]}
        for p in r["paragraphs"]:
            t=norm(p["title"]);b=norm(p["paragraph_text"])
            titles.add(t);body.add(b)
            if p["idx"] in supports:
                sup_titles.add(t);sup_body.add(b)
    prev_path=Path(os.environ["PREVIOUS_HOTPOT"])
    hot={json.loads(l)["id"] for l in prev_path.open()}
    gold=[[r["bag"][j] for j in gold_indices] for r,gold_indices in zip(inputs,idx)]
    result=dict(claims=len(inputs),prior_musique_train_questions=len(train["indices"]),
                claims_exactly_equal_normalized_musique_training_question=sum(norm(r["claim"]) in qs for r in inputs),
                earlier_hotpot_evaluation_file_distinct_ids=len(hot),
                claims_with_antecedent_in_earlier_hotpot_evaluation=sum(k in hot for k in cluster),
                distinct_antecedents_in_earlier_hotpot_evaluation=len(set(cluster)&hot),
                shuffle_or_tune_from_overlap_audit=False,
                source_hashes=dict(musique_train=sha(source),split=sha(splitpath),earlier_hotpot_examples=sha(prev_path)))
    for name,field,reference in [("candidate_title","title",titles),("support_title","title",sup_titles),
                                 ("candidate_body","text",body),("support_body","text",sup_body)]:
        result["claims_with_any_gold_matching_musique_training_"+name]=sum(
            any(norm(p[field]) in reference for p in bag) for bag in gold)
    put("overlap_receipt.json",result)
    print(result)


if __name__=="__main__":main()
