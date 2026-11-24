"""Score a predeclared target only after every checkpoint is frozen."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from train_scorers import ROOT, OUT, P, OPT, sha, data_for, take, construct, infer, metrics
from frozen_model import build_model as build_raw_model


def strata_for(target, d):
    ids = d["ids"].astype(str)
    masks = {"all":np.ones(len(ids),bool)}
    if target == "hotpot":
        for kind in ["bridge","comparison"]:
            masks[kind] = d["types"] == kind
        for name, filename in [
            ("clean_all_sources","clean_all_sources_selection_ids.json"),
            ("clean_musique_train","clean_musique_train_selection_ids.json"),
            ("clean_previously_seen_eval","clean_previously_seen_eval_selection_ids.json")]:
            selected = set(json.loads((ROOT/"data/hotpot"/filename).read_text()))
            masks[name] = np.asarray([i in selected for i in ids])
    if target == "2wiki":
        for kind in ["compositional", "inference"]:
            masks[kind] = d["types"] == kind
        for name, filename in [("clean_musique_train","clean_musique_train_selection_ids.json"),("clean_previous_eval","clean_previous_eval_selection_ids.json"),("clean_all_sources","clean_all_sources_selection_ids.json")]:
            selected=set(json.loads((ROOT/"data/wiki_transfer"/filename).read_text()))
            masks[name]=np.asarray([i in selected for i in ids])
    if target == "musique":
        old = json.loads((ROOT/"results/exposure/unexposed_subset_ids.json").read_text())
        key = "ids" if "ids" in old else "subset_ids"
        selected = set(old[key])
        masks["no_support_candidate_exposure_original_train"] = np.asarray([i in selected for i in ids])
    return masks


def report(pred, d, masks, ordered):
    return {name:metrics({k:v[mask] for k,v in pred.items()}, d["gold_original"][mask], ordered=ordered)
            for name,mask in masks.items()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target",choices=["lockbox","musique","hotpot","2wiki"],required=True)
    args=ap.parse_args()
    target=args.target
    frozen=json.loads((OUT/"models_frozen.json").read_text())
    assert frozen["status"]=="frozen" and len(frozen["checkpoint_sha256"])==100
    for file,digest in {**frozen["lock"], **frozen["checkpoint_sha256"]}.items():
        assert sha(ROOT/file)==digest,file
    stage="selection" if target=="lockbox" else "refit"
    dataset="musique" if target=="lockbox" else target
    split="train" if target=="lockbox" else "dev"
    base=ROOT/("data/transfer_contexts" if dataset=="2wiki" else "data/contexts")/dataset
    paths=["scripts/evaluate_scorers.py","scripts/train_scorers.py","scripts/bounded_model.py","scripts/frozen_model.py",
           "protocol.json","results/scorers/models_frozen.json",
           str((base/f"indices_{split}.npz").relative_to(ROOT)).replace("\\","/"),
           str((base/f"encoder_manifest_{split}.json").relative_to(ROOT)).replace("\\","/")]
    if target=="hotpot":
        paths += ["scripts/encode_hotpot_context.py","data/contexts/hotpot/adapter_lock.json",
                  "data/hotpot/manifest.json","data/hotpot/selection_ids.json",
                  "data/hotpot/clean_all_sources_selection_ids.json",
                  "data/hotpot/clean_musique_train_selection_ids.json",
                  "data/hotpot/clean_previously_seen_eval_selection_ids.json"]
    elif target=="2wiki":
        paths += ["scripts/encode_wiki_context.py","data/wiki_transfer/manifest.json","data/wiki_transfer/selection_ids.json", "data/wiki_transfer/clean_musique_train_selection_ids.json", "data/wiki_transfer/clean_previous_eval_selection_ids.json", "data/wiki_transfer/clean_all_sources_selection_ids.json"]
    elif target=="musique":
        paths += ["results/exposure/unexposed_subset_ids.json"]
    else:
        paths += ["results/splits/split_ids.json"]
    lock={p:sha(ROOT/p) for p in paths}
    folder=OUT/f"evaluation_{target}"
    folder.mkdir(exist_ok=True)
    if (folder/"evaluation_lock.json").exists():
        assert json.loads((folder/"evaluation_lock.json").read_text())==lock
    else:
        (folder/"evaluation_lock.json").write_text(json.dumps(lock,indent=2))
    feature_manifest=json.loads((base/f"encoder_manifest_{split}.json").read_text())
    assert sha(base/f"context_{split}.npy")==feature_manifest["context_file_sha256"]
    start=time.time()
    results=[]
    def get_data(seed):
        d=data_for(dataset,split,seed)
        if target=="lockbox":
            selected=json.loads((ROOT/"results/splits/split_ids.json").read_text())["lockbox_ids"]
            lookup={str(x):i for i,x in enumerate(d["ids"])}
            d=take(d,np.asarray([lookup[x] for x in selected]))
        return d
    for seed in OPT["seeds"]:
        d=get_data(seed)
        masks=strata_for(target,d)
        if target in ("hotpot","2wiki"):
            source="data/hotpot" if target=="hotpot" else "data/wiki_transfer"
            assert d["ids"].tolist()==json.loads((ROOT/source/"selection_ids.json").read_text())
        for condition in P["conditions"]:
            name=f"s{seed}_{condition['name']}"
            checkpoint=OUT/stage/f"{name}.pt"
            saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
            model=construct(condition,seed,saved["record"]["residual_cap"])
            model.load_state_dict(saved["state_dict"])
            pred=infer(model,d,condition,interventions=True)
            scores=report(pred,d,masks,ordered=target!="hotpot")
            record={"condition":condition["name"],"seed":seed,"checkpoint":checkpoint.relative_to(ROOT).as_posix(),
                    "checkpoint_sha256":sha(checkpoint),"metrics":scores["all"],"strata":scores}
            np.savez_compressed(folder/f"{name}_predictions.npz", ids=d["ids"],gold_original=d["gold_original"],
                                **pred)
            results.append(record)
            print(json.dumps({"target":target,**record}),flush=True)
        del d
        torch.cuda.empty_cache()
    d=get_data(0)
    masks=strata_for(target,d)
    baselines=[]
    for name,source,kind in [("frozen_e5_direct","question","direct"),
                              ("frozen_e5_context","context","chain"),
                              ("frozen_e5_passage","passage","chain")]:
        model=build_raw_model(kind=kind,input_dim=768,projection_dim=768,logit_scale=20.).cuda()
        with torch.no_grad():
            model.query_projection.weight.copy_(torch.eye(768,device="cuda"))
            model.key_projection.weight.copy_(torch.eye(768,device="cuda"))
        condition={"name":name,"source":source,"kind":kind}
        pred=infer(model,d,condition)
        scores=report(pred,d,masks,ordered=target!="hotpot")
        np.savez_compressed(folder/f"{name}_predictions.npz",ids=d["ids"],gold_original=d["gold_original"],**pred)
        baselines.append({"condition":name,"seed":None,"metrics":scores["all"],"strata":scores})
    (folder/"all_results.json").write_text(json.dumps(results,indent=2))
    (folder/"baselines.json").write_text(json.dumps(baselines,indent=2))
    (folder/"manifest.json").write_text(json.dumps({
        "status":"complete","target":target,"model_stage":stage,"model_records":len(results),
        "baseline_records":len(baselines),"n":len(d["ids"]),"strata":{k:int(v.sum()) for k,v in masks.items()},
        "seconds":time.time()-start,"lock":lock,"models_frozen_time":frozen["time"],
        "outcome_based_selection":False,"gold_order_used":target!="hotpot",
        "evaluation_scope":"New 2Wiki instances from seen distribution" if target=="2wiki" else "Previously examined target; exploratory follow-up",
        "direct_ordered_em":"not reported for top2; distinct ordered decoder reported separately"},indent=2))
    print("EVALUATION COMPLETE",target,flush=True)


if __name__=="__main__":
    main()
