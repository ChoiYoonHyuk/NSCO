"""Fixed bounded-adapter grid; freeze all models before evaluation."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from bounded_model import build_model, loss_terms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/scorers"
P = json.loads((ROOT / "protocol.json").read_text())
OPT = P["optimization"]
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def data_for(dataset, split, seed):
    directory = ROOT / ("data/transfer_contexts" if dataset == "2wiki" else "data/contexts") / dataset
    idx = dict(np.load(directory / f"indices_{split}.npz"))
    feature_path = ROOT / "data/e5/features.npy" if dataset == "musique" else directory / "features.npy"
    f = np.load(feature_path, mmap_mode="r")
    c = np.load(directory / f"context_{split}.npy", mmap_mode="r")
    n, width = idx["p"].shape
    rng = np.random.default_rng(seed + 989)
    permutations = np.stack([rng.permutation(width) for _ in range(n)])
    inverse = np.argsort(permutations, axis=1)
    labels = np.take_along_axis(inverse, idx["labels"][:, :2], axis=1)
    random_original = np.random.default_rng(seed + 782).integers(0, idx["candidate_count"])
    random_labels = inverse[np.arange(n), random_original]
    d = {
        "q": torch.from_numpy(np.array(f[idx["q"]])).cuda(),
        "p": torch.from_numpy(np.array(f[np.take_along_axis(idx["p"], permutations, axis=1)])).cuda(),
        "context": torch.from_numpy(np.array(c[np.arange(n)[:, None], permutations])).cuda(),
        "mask": torch.from_numpy(permutations < idx["candidate_count"][:, None]).cuda(),
        "labels": torch.from_numpy(labels).long().cuda(),
        "random": torch.from_numpy(random_labels).long().cuda(),
        "ids": idx["ids"], "permutations": permutations,
        "gold_original": idx["labels"][:, :2],
    }
    if "types" in idx:
        d["types"] = idx["types"]
    return d


def take(d, ix):
    return {k: v[ix] for k, v in d.items()}


def construct(condition, seed, cap):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return build_model(kind=condition["kind"], residual_cap=cap, **P["model"]).cuda()


def forward(model, d, condition, intervention=False):
    source = condition["source"]
    c = (d["q"][:, None, :].expand_as(d["p"]) if intervention or source == "question"
         else d["context"] if source == "context" else None)
    return model(d["q"], d["p"], d["mask"], c)


def decode_distinct(q, p, mask, terminal_scores):
    first = 20*torch.einsum("bd,bnd->bn", q, p)
    first = F.log_softmax(first.masked_fill(~mask, -torch.inf), -1)
    terminal = F.log_softmax(terminal_scores.masked_fill(~mask, -torch.inf), -1)
    width = p.shape[1]
    valid = mask[:, :, None] & mask[:, None, :] & ~torch.eye(width, dtype=torch.bool, device=q.device)[None]
    joint = (first[:, :, None]+terminal[:, None, :]).masked_fill(~valid, -torch.inf)
    joint = joint - torch.logsumexp(joint.flatten(1), -1)[:, None, None]
    marginal = torch.logsumexp(joint, 1).argmax(-1)
    flat = joint.flatten(1).argmax(-1)
    unordered = torch.logaddexp(joint, joint.transpose(1,2))
    unordered = unordered.masked_fill(~torch.triu(valid, diagonal=1), -torch.inf).flatten(1).argmax(-1)
    return marginal, torch.stack([flat//width, flat%width], -1), torch.stack([unordered//width, unordered%width], -1)


@torch.no_grad()
def infer(model, d, condition, interventions=False):
    model.eval()
    collected = {}
    for start in range(0, len(d["ids"]), 256):
        b = take(d, slice(start, start+256))
        o = forward(model, b, condition)
        direct = condition["kind"] == "direct"
        result = {"first": o["first_logits"].argmax(-1),
                  "terminal": o["terminal_logits"].argmax(-1),
                  "ordered": o["direct_top2_indices"] if direct else o["joint_map_indices"],
                  "pair": o["direct_top2_indices"] if direct else o["unordered_pair_indices"]}
        if direct:
            a, z, u = decode_distinct(b["q"], b["p"], b["mask"], o["direct_logits"])
            result.update(distinct_terminal=a, distinct_ordered=z, distinct_pair=u)
        if interventions and condition["source"] == "context":
            z = forward(model, b, condition, intervention=True)
            result.update(no_context_terminal=z["terminal_logits"].argmax(-1),
                          no_context_pair=z["unordered_pair_indices"])
        for k, v in result.items():
            original = np.take_along_axis(b["permutations"], v.cpu().numpy().reshape(len(v), -1), axis=1)
            if v.ndim == 1:
                original = original[:, 0]
            collected.setdefault(k, []).append(original)
    return {k: np.concatenate(v) for k, v in collected.items()}


def metrics(pred, gold, ordered=True):
    result = {}
    if ordered:
        result.update(terminal_r1=float((pred["terminal"] == gold[:,1]).mean()),
                      first_r1=float((pred["first"] == gold[:,0]).mean()))
        if "distinct_ordered" in pred:
            result["distinct_ordered_em"] = float((pred["distinct_ordered"] == gold).all(1).mean())
        else:
            result["ordered_em"] = float((pred["ordered"] == gold).all(1).mean())
    intersection = (pred["pair"][:, :, None] == gold[:, None, :]).any(-1).sum(1)
    result.update(support_em=float((intersection == 2).mean()),
                  support_f1=float((intersection/2).mean()), n=len(gold))
    for prefix in ["distinct", "no_context"]:
        if prefix+"_pair" in pred:
            pair = pred[prefix+"_pair"]
            overlap = (pair[:,:,None] == gold[:,None,:]).any(-1).sum(1)
            result[prefix+"_support_em"] = float((overlap == 2).mean())
            result[prefix+"_support_f1"] = float((overlap/2).mean())
            if ordered:
                result[prefix+"_terminal_r1"] = float((pred[prefix+"_terminal"] == gold[:,1]).mean())
    return result


def fit(train, tune, condition, seed, lr, cap, steps, path, source=None):
    model = construct(condition, seed, cap)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=OPT["weight_decay"])
    generator = torch.Generator().manual_seed(seed+1234)
    best, selected, state, curve = -1., 0, None, []
    start = time.time()
    for step in range(steps+1):
        if step:
            model.train()
            ix = torch.randint(len(train["ids"]), (OPT["batch_size"],), generator=generator).cuda()
            b = {k: v[ix] for k,v in train.items() if torch.is_tensor(v)}
            output = forward(model, b, condition)
            terms = loss_terms(output, b["labels"][:,1], b["labels"][:,0],
                               condition=condition["objective"], random_target=b["random"])
            loss = terms["loss"]
            assert torch.isfinite(loss), (condition["name"], seed, step)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), OPT["gradient_clip"])
            assert torch.isfinite(norm), (condition["name"], seed, step, "gradient")
            optimizer.step()
        if tune is not None and step in OPT["checkpoints"]:
            m = metrics(infer(model,tune,condition), tune["gold_original"])
            curve.append({"step":step, **m})
            if m["terminal_r1"] > best:
                best, selected = m["terminal_r1"], step
                state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if tune is None:
        selected = steps
        state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    record = {"condition":condition, "seed":seed, "lr":lr, "residual_cap":cap, "fit_steps":steps,
              "selected_step":selected, "selection_terminal_r1":best if tune is not None else None,
              "n_train":len(train["ids"]), "parameters":sum(p.numel() for p in model.parameters()),
              "curve":curve, "source":source, "seconds":time.time()-start}
    torch.save({"state_dict":state,"record":record},path)
    Path(str(path)+".json").write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k not in ["curve","source"]}),flush=True)
    return record


def training():
    start = time.time()
    OUT.mkdir(exist_ok=True)
    names = ["selected_settings.json","protocol.json","scripts/bounded_model.py","scripts/train_scorers.py","scripts/frozen_model.py",
             "scripts/encode_context.py","results/splits/split_ids.json",
             "data/contexts/musique/encoder_manifest_train.json",
             "data/wiki_transfer/manifest.json","data/wiki_transfer/selection_ids.json",
             "data/wiki_transfer/clean_all_sources_selection_ids.json"]
    lock = {n:sha(ROOT/n) for n in names}
    lock_path = OUT/"execution_lock.json"
    if lock_path.exists():
        assert json.loads(lock_path.read_text()) == lock
    else:
        lock_path.write_text(json.dumps(lock,indent=2))
    split = json.loads((ROOT/"results/splits/split_ids.json").read_text())
    idx = dict(np.load(ROOT/"data/contexts/musique/indices_train.npz"))
    lookup = {str(x):i for i,x in enumerate(idx["ids"])}
    ix = {k:np.array([lookup[x] for x in split[k]]) for k in ["train_ids","tune_ids","lockbox_ids","full_refit_ids"]}
    assert len(ix["full_refit_ids"]) == P["data"]["eligible_train"]
    for name in ["selection","refit"]:
        (OUT/name).mkdir(exist_ok=True)
    tuning = json.loads((ROOT / "selected_settings.json").read_text())
    for seed in OPT["seeds"]:
        d=data_for("musique","train",seed)
        train,tune=take(d,ix["train_ids"]),take(d,ix["tune_ids"])
        for condition in P["conditions"]:
            path=OUT/"selection"/f"s{seed}_{condition['name']}.pt"
            if not path.exists():
                fit(train,tune,condition,seed,tuning[condition["name"]]["lr"],tuning[condition["name"]]["residual_cap"],OPT["max_steps"],path)
        del train,tune
        full=take(d,ix["full_refit_ids"])
        for condition in P["conditions"]:
            path=OUT/"refit"/f"s{seed}_{condition['name']}.pt"
            source=json.loads((OUT/"selection"/f"s{seed}_{condition['name']}.pt.json").read_text())
            steps=math.ceil(source["selected_step"]*len(full["ids"])/len(ix["train_ids"]))
            if not path.exists():
                fit(full,None,condition,seed,tuning[condition["name"]]["lr"],tuning[condition["name"]]["residual_cap"],steps,path,source=source)
        del d,full
        torch.cuda.empty_cache()
    checkpoints={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for directory in ["selection","refit"]
                 for p in (OUT/directory).glob("*.pt")}
    assert len(checkpoints)==100
    (OUT/"models_frozen.json").write_text(json.dumps({"status":"frozen","time":time.time(),"lock":lock,
                                                     "checkpoint_sha256":checkpoints},indent=2))
    (OUT/"run_manifest.json").write_text(json.dumps({"status":"trained","seconds":time.time()-start,
                                                     "selection_models":50,"refit_models":50,"preselected_settings":len(tuning),
                                                     "lock":lock,"tf32":False},indent=2))
    print("ALL MODELS FROZEN; no test outcomes evaluated",flush=True)


if __name__=="__main__":
    training()
