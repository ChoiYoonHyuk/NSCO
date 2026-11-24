"""Fixed-rate confirmation of the Gaussian parity Adam crossing in Appendix S."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time

import torch
import gaussian as baseline


SEEDS = (711201, 711217, 711229, 711241, 711253)
RATES = (0.003, 0.02)
ARMS = ("final", "value", "index")
OUT = Path(__file__).resolve().parent / "results"


def write_rows(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def execute_seed(seed):
    # Baseline cell order is task, supervision, optimizer, initialization.
    configs = list(itertools.product(RATES, range(len(ARMS))))
    cells = [(2, sup, 1, 1) for _, sup in configs]
    frames, ids = baseline.configs(seed, cells)
    teacher = baseline.frame(seed)
    test = baseline.draw(baseline.TEST, baseline.generator(seed, 5), teacher)
    train_rng = baseline.generator(seed, 3)
    rate = torch.tensor([lr for lr, _ in configs], device=baseline.DEVICE)[:, None, None]
    first_moment = torch.zeros_like(frames)
    second_moment = torch.zeros_like(frames)
    rows = []
    for step in range(1, baseline.GRID[-1] // baseline.BATCH + 1):
        batch = baseline.draw(baseline.BATCH, train_rng, teacher)
        grad = baseline.gradient(frames, ids, batch)
        with torch.no_grad():
            # Original Adam update and schedule; each example appears in one minibatch.
            grad /= grad.norm(dim=(-2, -1), keepdim=True).clamp_min(1.)
            first_moment.mul_(.9).add_(grad, alpha=.1)
            second_moment.mul_(.999).addcmul_(grad, grad, value=.001)
            direction = (first_moment / (1 - .9 ** step)) / (
                (second_moment / (1 - .999 ** step)).sqrt() + 1e-8
            )
            frames -= (rate / math.sqrt(1 + (step - 1) / 128.)) * direction
        n = step * baseline.BATCH
        if n not in baseline.GRID:
            continue
        metrics = baseline.evaluate(baseline.polar(frames.detach()), ids[0], teacher, test)
        for (lr, sup), met in zip(configs, metrics):
            rows.append(dict(seed=seed, supervision=ARMS[sup], rate=lr, n=n,
                             clean_error=met["error"], whole_path=met["path"]))
    return rows


def summarize(rows):
    seeds = []
    for seed, lr, arm in itertools.product(SEEDS, RATES, ARMS):
        group = [r for r in rows if r["seed"] == seed and r["rate"] == lr
                 and r["supervision"] == arm]
        assert [r["n"] for r in group] == baseline.GRID
        passing = [r["n"] for r in group if r["clean_error"] <= .10]
        by_n = {r["n"]: r for r in group}
        seeds.append(dict(seed=seed, supervision=arm, rate=lr,
                          crossing=passing[0] if passing else "",
                          error_32768=by_n[32768]["clean_error"],
                          error_131072=by_n[131072]["clean_error"]))
    groups = []
    for lr, arm in itertools.product(RATES, ARMS):
        group = [r for r in seeds if r["rate"] == lr and r["supervision"] == arm]
        costs = sorted(r["crossing"] for r in group if isinstance(r["crossing"], int))
        groups.append(dict(rate=lr, supervision=arm, seeds=len(group), reached=len(costs),
                           median_crossing=costs[len(group)//2] if len(costs)>len(group)//2 else "",
                           mean_error_32768=statistics.mean(r["error_32768"] for r in group),
                           mean_error_131072=statistics.mean(r["error_131072"] for r in group)))
    return seeds, groups


def main():
    assert set(SEEDS).isdisjoint([baseline.DEV, *baseline.CONF])
    OUT.mkdir(exist_ok=True)
    protocol = dict(seeds=SEEDS, fixed_rates=RATES, arms=ARMS, batch=baseline.BATCH,
                    max_n=baseline.GRID[-1], grid=baseline.GRID, test=baseline.TEST,
                    theta=baseline.THETA, device=baseline.DEVICE, torch=str(torch.__version__),
                    initialization="small", optimizer="adam", task="Parity",
                    summary="median first tested n with clean polarized error <= .10; censored last; mean errors at 32768/131072",
                    helper_sha256=hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / "fixed_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    for seed in SEEDS:
        start = time.monotonic()
        rows.extend(execute_seed(seed))
        print(f"done {seed} in {time.monotonic()-start:.1f}s", flush=True)
    seeds, groups = summarize(rows)
    write_rows("fixed_curves.csv", rows)
    write_rows("fixed_seeds.csv", seeds)
    write_rows("fixed_summary.csv", groups)
    print(json.dumps(groups, indent=2), flush=True)


if __name__ == "__main__":
    main()
