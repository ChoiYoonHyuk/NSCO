"""Audit every planned cell and calculate teacher-level paired statistics."""
import csv
import itertools
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "plan.json").read_text())
RESULTS = ROOT / "results"


def write(name, rows):
    with (RESULTS / name).open("w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_curves():
    rows = []
    for seed, depth in itertools.product(PLAN["seeds"], PLAN["depths"]):
        with (RESULTS / f"metrics_{seed}_{depth}.csv").open() as f:
            group = list(csv.DictReader(f))
        for r in group:
            for key in ("seed", "depth", "n", "labels", "errors", "path_hits", "test"):
                r[key] = int(r[key])
            r["rate"] = float(r["rate"])
            assert r["seed"] == seed and r["depth"] == depth
            assert r["labels"] == r["n"] * (1 + (r["arm"] == "value"))
            assert r["test"] == PLAN["test_examples"]
            assert 0 <= r["errors"] <= r["test"] and 0 <= r["path_hits"] <= r["test"]
        actual = [(r["rate"], r["arm"], r["n"]) for r in group]
        expected = itertools.product(PLAN["rates"], PLAN["arms"], PLAN["grid_examples"])
        assert sorted(actual) == sorted(expected)
        rows.extend(group)
    return rows


def quantile(ordered, q):
    position = (len(ordered)-1) * q
    a, f = divmod(position, 1)
    i = int(a)
    return ordered[i] + f * (ordered[min(i+1, len(ordered)-1)]-ordered[i])


def paired_interval(deltas):
    rng = random.Random(721001)
    means = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(50000))
    return quantile(means, .025), quantile(means, .975)


def sign_test(deltas):
    positive = sum(d > 0 for d in deltas)
    negative = sum(d < 0 for d in deltas)
    n = positive + negative
    return min(1., 2*sum(math.comb(n, j) for j in range(min(positive, negative)+1))/2**n)


def report():
    rows = read_curves()
    index = {(r["seed"], r["depth"], r["rate"], r["arm"], r["n"]): r for r in rows}
    summaries, endpoints, by_seed, crosses = [], [], [], []
    for depth, rate in itertools.product(PLAN["depths"], PLAN["rates"]):
        for budget in ("labels", "n"):
            finals, auxs = [], []
            for seed in PLAN["seeds"]:
                fn = PLAN["max_examples"]
                an = fn // (2 if budget == "labels" else 1)
                pair = [index[seed, depth, rate, a, n] for a,n in (("final", fn), ("value", an))]
                f, a = [100*r["errors"]/r["test"] for r in pair]
                endpoints.append(dict(depth=depth, rate=rate, budget_unit=budget,
                                      seed=seed, final_error_pct=f, value_error_pct=a,
                                      delta_pp=a-f))
                finals.append(f)
                auxs.append(a)
            deltas = [a-f for f,a in zip(finals, auxs)]
            lo, hi = paired_interval(deltas)
            summaries.append(dict(depth=depth, rate=rate, budget_unit=budget, budget=fn,
                                  final_error_pct=statistics.fmean(finals),
                                  value_error_pct=statistics.fmean(auxs),
                                  delta_pp=statistics.fmean(deltas), ci95_low=lo, ci95_high=hi,
                                  value_better=sum(d<0 for d in deltas), seeds=len(deltas),
                                  sign_test_two_sided=sign_test(deltas)))
            for arm in PLAN["arms"]:
                first = []
                for seed in PLAN["seeds"]:
                    curve = [index[seed, depth, rate, arm, n] for n in PLAN["grid_examples"]]
                    within = [r for r in curve if r[budget] <= fn]
                    t = next((r[budget] for r in within if r["errors"]*10 <= r["test"]), None)
                    first.append(t if t is not None else math.inf)
                    by_seed.append(dict(seed=seed, depth=depth, rate=rate, arm=arm, budget_unit=budget,
                                        budget=fn, first_crossing=t, last_test=within[-1][budget]))
                first.sort()
                mid = len(first)//2
                crosses.append(dict(depth=depth, rate=rate, arm=arm, budget_unit=budget,
                                    budget=fn, reached=sum(math.isfinite(x) for x in first),
                                    seeds=len(first), median_lower=first[mid-1], median_upper=first[mid]))
    write("endpoint_summary.csv", summaries)
    write("endpoint_pairs.csv", endpoints)
    write("crossing_summary.csv", crosses)
    write("crossing_by_teacher.csv", by_seed)


if __name__ == "__main__":
    report()
