"""Lock unscored remaining directed 2Wiki instances using metadata only.

Reuses the audited graph rules and archived official dev file. Excludes the
prior 2,000 evaluated IDs and exact/QA-normalized known question strings, then
fixes all remaining IDs and source-specific support-text exposure strata.
No encoder/model/prediction/metric is loaded and no new download is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from prepare_wiki_reference import derive, independent_audit
from prepare_hotpot import (sha, write_json, lock_json, jsonl, empty_reference,
                               add_candidate, overlap_flags, normalized_question)

ROOT = Path(__file__).resolve().parents[1]


def load_reference(path, kind):
    ref, n, nc = empty_reference(), 0, 0
    exact_questions = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line); n += 1
            exact_questions.add(row["question"])
            ref["questions"].add(normalized_question(row["question"]))
            if kind == "musique":
                for step in row["question_decomposition"]:
                    exact_questions.add(step["question"])
                    ref["component_questions"].add(normalized_question(step["question"]))
            paragraphs = ([(p["title"],p["text"]) for p in row["candidates"]] if kind == "hotpot" else
                          [(p["title"],p["paragraph_text"]) for p in row["paragraphs"]])
            for title, text in paragraphs:
                add_candidate(ref,title,text); nc += 1
    return ref, exact_questions, {"path":str(path),"sha256":sha(path),"rows":n,
        "actual_candidate_paragraph_occurrences":nc,
        "unique_reference_counts":{k:len(v) for k,v in ref.items()},
        "scope":"All actual candidates, including distractors; raw/normalized composed and available constituent question strings"}


def source_example(derived):
    return {"id":derived["id"],"question":derived["question"],
            "candidates":[{"title":p["title"],"text":p["paragraph_text"]} for p in derived["paragraphs"]],
            "gold_support_indices":derived["support_labels"],"type":derived["type"]}


def self_test():
    row={"_id":"x","question":"Who?","type":"inference","answer":"C",
         "context":[["B",["body b"]],["A",["body a"]],["D",["body d"]]],
         "supporting_facts":[["A",0],["B",0]],"evidences":[["B","r","C"],["A","r","B"]]}
    example, reason=derive(row)
    assert reason=="eligible" and example["support_labels"]==[1,0]
    independent_audit(example,row)
    changed=dict(row,evidences=list(reversed(row["evidences"])))
    assert source_example(derive(changed)[0])==source_example(example)
    x=source_example(example)
    assert set(x)=={"id","question","candidates","gold_support_indices","type"}
    assert [p["title"] for p in x["candidates"]]==["B","A","D"]
    assert "answer" not in x and "evidences" not in x
    print("Self-test passed: directed labels independent of evidence order, candidate order preserved, target metadata separated")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT);parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test: self_test();return
    root=args.root.resolve();old=root/"data/wiki_reference";out=root/"data/wiki_transfer"
    out.mkdir(parents=True,exist_ok=True)
    prior=json.loads((old/"manifest.json").read_text(encoding="utf-8"))
    rawpath=old/"official_dev.json"
    assert sha(rawpath)==prior["source"]["dev_sha256"]
    raw=json.loads(rawpath.read_text(encoding="utf-8"))
    prior_ids=json.loads((old/"selection_ids.json").read_text(encoding="utf-8"))
    assert len(prior_ids)==len(set(prior_ids))==2000
    assert sha(old/"selection_ids.json")==prior["selected_ids_sha256"]
    previous=set(prior_ids);reasons=Counter();eligible={};sources={}
    for row in raw:
        assert row["_id"] not in sources;sources[row["_id"]]=row
        derived,reason=derive(row);reasons[reason]+=1
        if derived is not None:
            independent_audit(derived,row)
            assert derived["order_derivation"]["answer_exactly_matches_path_tail"]
            eligible[derived["id"]]=derived
    assert len(raw)==12576 and len(eligible)==3563 and previous<=set(eligible)
    assert dict(reasons)==prior["first_failure_counts"]
    ordered_ids=sorted(eligible,key=lambda x:(hashlib.sha256(x.encode()).hexdigest(),x))
    assert ordered_ids[:2000]==prior_ids
    remaining=[i for i in ordered_ids if i not in previous]
    assert len(remaining)==1563 and not previous.intersection(remaining)
    candidates=[source_example(eligible[i]) for i in remaining]
    paths={"musique_train":(root/"data/musique_ans_v1.0_train.jsonl","musique",19938),
           "previously_seen_musique_dev":(root/"data/musique_ans_v1.0_dev.jsonl","musique",2417),
           "previously_seen_2wiki_selected":(old/"examples.jsonl","2wiki",2000),
           "previously_seen_hotpot_dev":(root/"data/hotpot/examples.jsonl","hotpot",7405)}
    flags={i:{} for i in remaining};reports={}
    for name,(path,kind,expected_n) in paths.items():
        ref,exact,report=load_reference(path,kind)
        assert report["rows"]==expected_n
        for example in candidates:
            values=overlap_flags(example,ref)
            values["raw_question_exact"]=example["question"] in exact
            if values["raw_question_exact"]: values["clean_question_and_support_text"]=False
            flags[example["id"]][name]=values
        report["remaining_1563_example_overlap_counts"]={k:sum(flags[i][name][k] for i in remaining)
                                                       for k in flags[remaining[0]][name]}
        reports[name]=report
        del ref,exact
    question_keys=["raw_question_exact","normalized_question","normalized_constituent_question"]
    excluded=[{"id":i,"matches":{name:{k:flags[i][name][k] for k in question_keys}
                                   for name in paths if any(flags[i][name][k] for k in question_keys)}}
              for i in remaining if any(flags[i][name][k] for name in paths for k in question_keys)]
    excluded_ids={r["id"] for r in excluded}
    ids=[i for i in remaining if i not in excluded_ids]
    assert ids and not previous.intersection(ids)
    examples=[source_example(eligible[i]) for i in ids]
    annotations=[{"id":i,"answer":sources[i]["answer"],"supporting_facts":sources[i]["supporting_facts"],
                  "evidences":sources[i]["evidences"],"order_derivation":eligible[i]["order_derivation"]} for i in ids]
    lock_json(out/"selection_ids.json",ids)
    lock_json(out/"remaining_before_question_filter_ids.json",remaining)
    masks={}
    for name in paths:
        masks[f"clean_{name}_selection_ids.json"]=[i for i in ids if flags[i][name]["clean_question_and_support_text"]]
    seen=[name for name in paths if name!="musique_train"]
    masks["clean_previous_eval_selection_ids.json"]=[i for i in ids if all(flags[i][name]["clean_question_and_support_text"] for name in seen)]
    masks["clean_all_sources_selection_ids.json"]=[i for i in ids if all(flags[i][name]["clean_question_and_support_text"] for name in paths)]
    for name,selected in masks.items(): lock_json(out/name,selected)
    for name in paths:
        reports[name]["selected_example_overlap_counts"]={k:sum(flags[i][name][k] for i in ids)
                                                         for k in flags[ids[0]][name]}
    jsonl(out/"examples.jsonl",examples);jsonl(out/"annotations.jsonl",annotations)
    write_json(out/"known_question_exclusions.json",excluded)
    write_json(out/"overlap_by_id.json",{i:flags[i] for i in ids})
    overlap={"status":"Metadata only, before any external prediction; no outcome files read",
             "sources":reports,"remaining_before_question_filter":1563,"question_exclusions":len(excluded),
             "question_exclusion_rule":"Exclude raw exact or QA-normalized composed/available constituent question matches to any of the four known source sets",
             "normalization":"Questions: lowercase, ASCII punctuation removal, English article removal, whitespace collapse. Support texts: raw or whitespace-only normalized exact equality.",
             "clean_rule":"No selected gold support paragraph text matching ANY actual candidate paragraph in specified source(s). Exact titles and matches confined to target distractors are reported but not excluded.",
             "clean_counts":{name:len(selected) for name,selected in masks.items()},
             "limitations":"Unscored instances from the same previously examined 2Wiki dev distribution, not a novel external dataset or an unrestricted confirmatory test; shared Wikipedia, paraphrases and pretraining overlap remain possible."}
    write_json(out/"metadata_overlap.json",overlap)
    files=["selection_ids.json","remaining_before_question_filter_ids.json","examples.jsonl","annotations.jsonl",
           "known_question_exclusions.json","overlap_by_id.json","metadata_overlap.json",*masks]
    manifest={"status":"Preparation and fixed metadata strata complete; no external model outcomes",
              "created_utc":datetime.now(timezone.utc).isoformat(),
              "study":"Remaining unscored strict directed two-hop 2Wiki official-dev instances from an already examined source distribution",
              "source":prior["source"],"source_json":str(rawpath),"source_json_sha256":sha(rawpath),
              "official_repository":prior["official_repository"],"paper":prior["paper"],
              "prior_preparation_manifest_sha256":sha(old/"manifest.json"),
              "prior_selected_ids_sha256":sha(old/"selection_ids.json"),"prior_selected_ids_excluded":2000,
              "source_rows":12576,"eligible_strict_directed_twohop":3563,
              "remaining_before_question_filter":1563,"known_question_exclusions":len(excluded),"selected_count":len(ids),
              "eligible_first_failure_counts":dict(reasons),"selected_types":dict(Counter(x["type"] for x in examples)),
              "selected_candidate_counts":dict(Counter(len(x["candidates"]) for x in examples)),
              "selection_rule":"All eligible IDs outside the prior locked first-2000 and known question matches; ascending original SHA256(id), then ID; no outcome-based sampling",
              "candidate_order":"Original official context order preserved",
              "gold_order":"Directed [first, terminal] from unique two-edge evidence path; never supporting-facts/evidence list order",
              "graph_policy":prior["order_policy"],"independent_degree_audit":"passed for all3563eligible and all selected labels",
              "encoder_input_policy":"Only question and candidates[].title/text allowed; gold_support_indices/type are scoring metadata; answer/evidence/support sentence/order derivation stored separately in annotations.jsonl",
              "primary_clean_selection_file":"clean_all_sources_selection_ids.json","metadata_overlap":overlap,
              "files_sha256":{name:sha(out/name) for name in files},
              "script_sha256":sha(Path(__file__)),
              "helper_scripts_sha256":{name:sha(root/name) for name in ["scripts/prepare_wiki_reference.py","scripts/prepare_hotpot.py"]},
              "boundaries":"No train or test members parsed; no new download, encoder loading, model fitting, model selection, predictions, or outcomes. IDs/strata fixed before any transfer evaluation."}
    write_json(out/"manifest.json",manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
