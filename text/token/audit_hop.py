"""Re-evaluate checkpoint labels and hard learned routes with explicit index loops."""
import csv
import hashlib
import json
from pathlib import Path
import torch
import numpy as np
from native_check import HERE, SEEDS, InductionHopsFullSequenceTask, build_model, Munch, yaml
from sign_variant import ARMS, BUDGETS, LENGTH, Router, eval_records, records


def verify_native():
    folder = HERE / "upstream/models/induction_hops_full_seq/width4-depth6"
    conf = yaml.safe_load((folder / "config.yaml").read_text())
    model = build_model(Munch.fromDict(conf["model"]), 100).cuda().eval()
    model.load_state_dict(torch.load(folder / "model_100000.pt", weights_only=True, map_location="cpu"))
    given = {(int(r["seed"]), int(r["hops"])): r for r in csv.DictReader((HERE / "native_results.csv").open())}
    rows = []
    for seed in SEEDS:
        for h in (0, 3, 16):
            task = InductionHopsFullSequenceTask(100, 4, 0, 16, np.random.RandomState(seed * 100 + h))
            ids, labels = task.get_batch(128, hops=h)
            with torch.no_grad():
                logits = model(ids.cuda()).cpu().argmax(-1).numpy()
            ids, labels = ids.numpy(), labels.numpy()
            unique = len(set(map(tuple, ids)))
            assert unique == 128
            valid_counts = []
            errors = nonnull_errors = 0
            for i, row in enumerate(ids):
                chars = row[1:].tolist()
                valid = []
                for t in range(len(chars)):
                    p = t
                    for _ in range(h):
                        last = next((k for k in range(p-1, -1, -1) if chars[k] == chars[p]), None)
                        if last is None:
                            p = -1
                            break
                        p = last + 1
                    target = task.token_map[task.DOES_NOT_EXIST_CHAR] if p < 0 else chars[p]
                    assert labels[i, t+1] == target
                    if p >= 0:
                        valid.append(t+1)
                valid_counts.append(len(valid))
                wrong = logits[i] != labels[i]
                errors += int(wrong.sum())
                nonnull_errors += int(wrong[valid].sum())
            original = given[seed,h]
            observed = [errors, sum(valid_counts), nonnull_errors]
            published = [int(original[k]) for k in ("errors", "nonnull_char_tokens", "nonnull_char_errors")]
            rows.append(dict(seed=seed, hops=h, independent=observed, original=published,
                             nonnull_min=min(valid_counts), nonnull_max=max(valid_counts),
                             matched=observed==published, unique_sequences=unique,
                             ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
    return rows


def verify_routes():
    given = {(int(r["seed"]), r["readout"], r["supervision"]): r for r in
             csv.DictReader((HERE / "sign_results.csv").open()) if int(r["examples"]) == 8192}
    rows = []
    for seed in SEEDS:
        model = Router(seed).cuda()
        rng = np.random.RandomState(500000+seed)
        opt = torch.optim.Adam(model.parameters(), lr=.035)
        for _ in range(BUDGETS[-1]//32):
            opt.zero_grad()
            model(records(rng, 32)).backward()
            opt.step()
        ids, query, truth, signs = (x.cpu().numpy() for x in eval_records(seed))
        match = model.match.detach().cpu().numpy()
        decay = torch.nn.functional.softplus(model.decay.detach()).cpu().numpy()
        for j,(g,aux) in enumerate(ARMS):
            current = query.copy()
            b = np.arange(len(ids))
            inferred=[]
            for _ in range(3):
                candidate = np.arange(1,LENGTH)
                # position j stores the value after the earlier token at j-1.
                weight = match[j][ids[b,current,None], ids[:,:-1]] - decay[j]*np.log1p(np.maximum(0,current[:,None]-candidate))
                weight[candidate>=current[:,None]] = -np.inf
                current = np.where(current < 2, 0, 1+weight.argmax(1))
                inferred.append(current)
            path=np.stack(inferred,axis=1)
            read = {"first":lambda s:s[:,0],"last":lambda s:s[:,-1],"parity":lambda s:s.prod(1),
                    "majority":lambda s:np.sign(s.sum(1))}[g]
            values = signs[b[:,None], path]
            observed = [float(np.mean(read(values)!=read(signs[b[:,None],truth]))),
                        float(np.mean(path[:,0]==truth[:,0])),float(np.mean(np.all(path==truth,axis=1)))]
            original = given[seed,g,aux]
            published = [float(original[k]) for k in ("sign_error","first_accuracy","path_accuracy")]
            rows.append(dict(seed=seed, readout=g, supervision=aux, independent=observed,
                             original=published, matched=bool(np.allclose(observed,published,atol=1e-7,rtol=0))))
    return rows


if __name__ == "__main__":
    torch.set_num_threads(2)
    report = {"native":verify_native(),"sign":verify_routes()}
    (HERE/"independent_audit.json").write_text(json.dumps(report,indent=2))
    for k, group in report.items():
        print(k, sum(row["matched"] for row in group), "/", len(group), flush=True)
