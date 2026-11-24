"""Predeclared paired seed-by-example bootstrap; NumPy only, no model fitting.

The --self-test path never opens benchmark outcomes. The --run path requires
all 100 selected/refit checkpoints frozen and all four evaluations complete.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["lockbox", "musique", "hotpot", "2wiki"]
SEEDS = [11, 22, 33, 44, 55]
COMPARATORS = [
    ("context_terminal", "context_terminal", ""),
    ("frozen_e5_context", "frozen_e5_context", ""),
    ("direct_terminal_with_cosine_distinct", "direct_terminal", "distinct_"),
    ("context_teacher", "context_teacher", ""),
    ("context_random100", "context_random100", ""),
]
STRATA = {
    "lockbox": ["all"],
    "musique": ["all", "no_support_candidate_exposure_original_train"],
    "hotpot": ["all", "bridge", "comparison", "clean_all_sources",
               "clean_musique_train", "clean_previously_seen_eval"],
    "2wiki": ["all", "compositional", "inference", "clean_musique_train",
              "clean_previous_eval", "clean_all_sources"],
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def weighted_statistics(differences, seed_counts, example_counts):
    """Exact product-weight mean for K paired contrasts on S seeds and N items.

    differences is [K,S,N]; count matrices are [B,S] and [B,N]. Each row
    has S and N multinomial draws respectively. The same paired draws apply
    to every metric and comparator. No iid assumption over the S*N cells.
    """
    d = np.asarray(differences, dtype=np.float64)
    sw, ew = np.asarray(seed_counts), np.asarray(example_counts)
    assert d.ndim == 3 and sw.ndim == ew.ndim == 2
    k, s, n = d.shape
    assert sw.shape[1] == s and ew.shape == (sw.shape[0], n)
    assert np.all(sw >= 0) and np.all(ew >= 0)
    assert np.all(sw.sum(1) == s) and np.all(ew.sum(1) == n)
    by_seed = (ew @ d.reshape(k * s, n).T).reshape(len(ew), k, s) / n
    return np.einsum("bks,bs->bk", by_seed, sw) / s


def bootstrap(differences, repetitions, rng_seed, target_index, stratum_index, batch_size=100):
    """Two independent PCG64 streams per stratum; chunking is RNG invariant."""
    d = np.asarray(differences, dtype=np.float64)
    assert d.ndim == 3 and np.isfinite(d).all()
    _, s, n = d.shape
    assert s == 5 and n > 0 and repetitions > 0
    seed_rng = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([rng_seed, target_index, stratum_index, 0])))
    example_rng = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([rng_seed, target_index, stratum_index, 1])))
    # Draw all seed counts at once so chunking cannot alter their RNG stream.
    seed_counts = seed_rng.multinomial(s, np.full(s, 1.0 / s), size=repetitions)
    values = []
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        example_counts = example_rng.multinomial(n, np.full(n, 1.0 / n), size=stop - start)
        values.append(weighted_statistics(d, seed_counts[start:stop], example_counts))
    return np.concatenate(values, axis=0)


def metric_vectors(prediction, gold, counts, ordered, prefix=""):
    """Per-example scores on original candidate indices; two distinct golds."""
    gold = np.asarray(gold)
    counts = np.asarray(counts)
    n = len(gold)
    assert gold.shape == (n, 2) and counts.shape == (n,)
    assert np.issubdtype(gold.dtype, np.integer)
    assert np.all(gold[:, 0] != gold[:, 1])
    assert np.all((gold >= 0) & (gold < counts[:, None]))
    pair = np.asarray(prediction[prefix + "pair"])
    assert pair.shape == (n, 2) and np.issubdtype(pair.dtype, np.integer)
    assert np.all((pair >= 0) & (pair < counts[:, None]))
    assert np.all(pair[:, 0] != pair[:, 1])
    intersection = (pair[:, :, None] == gold[:, None, :]).any(-1).sum(1)
    result = {
        "support_em": (intersection == 2).astype(np.float64),
        "support_f1": intersection.astype(np.float64) / 2.0,
    }
    if ordered:
        terminal = np.asarray(prediction[prefix + "terminal"])
        assert terminal.shape == (n,) and np.issubdtype(terminal.dtype, np.integer)
        assert np.all((terminal >= 0) & (terminal < counts))
        result = {"terminal_r1": (terminal == gold[:, 1]).astype(np.float64), **result}
    return result


def self_test():
    """Independent deterministic fixtures; no project data is accessed."""
    checks = {}
    z = np.zeros((3, 5, 7))
    b = bootstrap(z, 251, 2027108, 0, 0, batch_size=37)
    assert np.array_equal(b, np.zeros((251, 3)))
    checks["constant_zero_distribution_and_interval"] = True
    a = np.arange(35, dtype=np.float64).reshape(5, 7) / 35
    same = bootstrap((a - a)[None], 251, 2027108, 0, 1)
    assert np.array_equal(same, np.zeros((251, 1)))
    checks["paired_identical_methods_zero"] = True
    constant = bootstrap(np.full((1, 5, 7), 0.25), 251, 2027108, 0, 2)
    assert np.array_equal(constant, np.full((251, 1), 0.25))
    checks["constant_nonzero_difference"] = True
    # Independently expand sampled indices, then average their Cartesian product.
    fixture = np.asarray([[[0, 1, -1], [0.5, 0, 1], [-1, 0.5, 0],
                           [1, -0.5, 0.5], [0, -1, 1]]], dtype=np.float64)
    sampled_seeds = [[0, 0, 2, 4, 4], [1, 2, 3, 3, 4]]
    sampled_examples = [[1, 1, 2], [0, 2, 2]]
    sw = np.asarray([np.bincount(x, minlength=5) for x in sampled_seeds])
    ew = np.asarray([np.bincount(x, minlength=3) for x in sampled_examples])
    manual = np.asarray([sum(fixture[0, s, e] for s in ss for e in ee) / 15
                         for ss, ee in zip(sampled_seeds, sampled_examples)])[:, None]
    np.testing.assert_allclose(weighted_statistics(fixture, sw, ew), manual, atol=1e-15, rtol=0)
    checks["manual_expanded_cartesian_weight_fixture"] = True
    one = bootstrap(fixture, 251, 2027108, 1, 3, batch_size=1)
    many = bootstrap(fixture, 251, 2027108, 1, 3, batch_size=83)
    np.testing.assert_array_equal(one, many)
    checks["batch_size_invariance"] = True
    gold = np.asarray([[0, 1], [1, 2], [1, 3]])
    p = {"pair": np.asarray([[1, 0], [2, 0], [0, 2]]), "terminal": np.asarray([1, 1, 3]),
         "distinct_pair": np.asarray([[0, 1], [1, 2], [3, 1]]),
         "distinct_terminal": np.asarray([0, 2, 3])}
    v = metric_vectors(p, gold, np.asarray([2, 3, 4]), ordered=True)
    np.testing.assert_array_equal(v["terminal_r1"], [1, 0, 1])
    np.testing.assert_array_equal(v["support_em"], [1, 0, 0])
    np.testing.assert_array_equal(v["support_f1"], [1, 0.5, 0])
    d = metric_vectors(p, gold, np.asarray([2, 3, 4]), ordered=True, prefix="distinct_")
    np.testing.assert_array_equal(d["support_em"], [1, 1, 1])
    assert "terminal_r1" not in metric_vectors(p, gold, np.asarray([2, 3, 4]), ordered=False)
    checks["metric_definitions_direct_distinct_and_unordered_hotpot"] = True
    raw = np.asarray([[1, 0, 1]])
    assert np.broadcast_to(raw, (5, 3)).mean() == raw.mean()
    checks["raw_control_seed_broadcast"] = True
    return {"status": "passed", "benchmark_outcomes_read": False, "checks": checks,
            "implementation_sha256": sha(__file__), "numpy_version": np.__version__}


def metadata_for(root, target, inputs):
    def track(relative):
        inputs[relative] = sha(root / relative)
        return root / relative
    dataset = "musique" if target == "lockbox" else target
    split = "train" if target == "lockbox" else "dev"
    base = "data/transfer_contexts" if target == "2wiki" else "data/contexts"
    relative = f"{base}/{dataset}/indices_{split}.npz"
    with np.load(track(relative), allow_pickle=False) as z:
        ids, gold, counts = z["ids"].astype(str), z["labels"][:, :2], z["candidate_count"]
        types = z["types"].astype(str) if "types" in z else None
    if target == "lockbox":
        selected = read_json(track("results/splits/split_ids.json"))["lockbox_ids"]
        lookup = {x: i for i, x in enumerate(ids)}
        ix = np.asarray([lookup[x] for x in selected])
        ids, gold, counts = ids[ix], gold[ix], counts[ix]
    masks = {"all": np.ones(len(ids), dtype=bool)}
    if target in ("hotpot", "2wiki"):
        source = "data/hotpot" if target == "hotpot" else "data/wiki_transfer"
        assert ids.tolist() == read_json(track(f"{source}/selection_ids.json"))
        for kind in (["bridge", "comparison"] if target == "hotpot" else ["compositional", "inference"]):
            masks[kind] = types == kind
        names = (["clean_all_sources", "clean_musique_train", "clean_previously_seen_eval"]
                 if target == "hotpot" else
                 ["clean_musique_train", "clean_previous_eval", "clean_all_sources"])
        for name in names:
            selected = set(read_json(track(f"{source}/{name}_selection_ids.json")))
            assert selected.issubset(set(ids))
            masks[name] = np.asarray([x in selected for x in ids])
    elif target == "musique":
        old = read_json(track("results/exposure/unexposed_subset_ids.json"))
        selected = set(old["ids"] if "ids" in old else old["subset_ids"])
        masks["no_support_candidate_exposure_original_train"] = np.asarray([x in selected for x in ids])
    assert list(masks) == STRATA[target]
    assert len(set(ids)) == len(ids) and all(mask.any() for mask in masks.values())
    return ids, gold, counts, masks


def tex_summary(records):
    """Six fixed groups, every comparator; no filtering by interval or sign."""
    groups = [(t, "all") for t in TARGETS] + [
        ("hotpot", "clean_all_sources"), ("2wiki", "clean_all_sources")]
    names = {"context_terminal": "Terminal only", "frozen_e5_context": "Frozen E5 context",
             "direct_terminal_with_cosine_distinct": "Direct, distinct decoder",
             "context_teacher": "Teacher forced", "context_random100": "Random auxiliary"}
    titles = {"lockbox": "MuSiQue model-specific lockbox (previously examined)",
              "musique": "MuSiQue official dev (previously examined)",
              "hotpot": "HotpotQA distractor dev (previously examined)",
              "2wiki": "New 2Wiki instances, previously studied distribution"}
    lines = [r"\begin{table*}[p]", r"\centering\small", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Comparator & Terminal R@1 & Set EM & Set F1 \\", r"\midrule"]
    for target, stratum in groups:
        selected = [r for r in records if r["target"] == target and r["stratum"] == stratum]
        n = selected[0]["n_examples"]
        suffix = "all" if stratum == "all" else "exact-question/support-text clean across all audited sources"
        lines.append(r"\multicolumn{4}{l}{\textit{" + titles[target] + f"; {suffix}; $n={n:,}$" + r"}} \\")
        for comparator, _, _ in COMPARATORS:
            cells = []
            for metric in ("terminal_r1", "support_em", "support_f1"):
                found = [r for r in selected if r["comparator"] == comparator and r["metric"] == metric]
                if not found:
                    assert target == "hotpot" and metric == "terminal_r1"
                    cells.append(r"---")
                else:
                    r = found[0]
                    low, high = r["percentile_95_ci_pp"]
                    cells.append(f"${r['difference_pp']:+.1f}\\,[{low:+.1f},{high:+.1f}]$")
            lines.append(names[comparator] + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{All prespecified primary bounded-residual contrasts: first-step auxiliary weight 1 minus each comparator, in percentage points. Brackets are approximate descriptive 95\% percentile intervals from 2,000 paired two-way bootstrap replicates: five training seeds and test examples are independently resampled, preserving method pairing in both dimensions. The frozen control is repeated across seed rows. No multiplicity adjustment is applied; five seeds provide limited information about training variability. HotpotQA has unordered gold supports, so terminal R@1 is undefined. Exact text filtering does not establish freedom from pretraining or broader Wikipedia overlap.}",
              r"\label{tab:two-way-bootstrap}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def run(root, protocol_path, outdir, tex_path):
    start = time.time()
    protocol = read_json(protocol_path)
    assert protocol["implementation_sha256"] == sha(__file__), "Implementation changed after protocol lock"
    assert protocol["rng_seed"] == 2027108 and protocol["repetitions"] == 2000
    assert protocol["seeds"] == SEEDS and protocol["target_strata"] == STRATA
    assert protocol["primary_condition"] == "context_first100"
    assert protocol["comparators"] == [x[0] for x in COMPARATORS]
    inputs = {str(protocol_path.relative_to(root)).replace("\\", "/"): sha(protocol_path),
              "scripts/bootstrap_scores.py": sha(__file__)}
    for relative, digest in protocol["release_source_sha256"].items():
        assert sha(root / relative) == digest, relative
        inputs[relative] = digest
    folder = root / "results/scorers"
    frozen_file = folder / "models_frozen.json"
    frozen = read_json(frozen_file)
    assert frozen["status"] == "frozen" and len(frozen["checkpoint_sha256"]) == 100
    assert protocol["created_unix_seconds"] < frozen["time"], "Protocol must precede completed-model boundary"
    inputs["results/scorers/models_frozen.json"] = sha(frozen_file)
    assert len(set(frozen["checkpoint_sha256"])) == 100
    for relative, digest in {**frozen["lock"], **frozen["checkpoint_sha256"]}.items():
        assert sha(root / relative) == digest, relative
        inputs[relative] = digest
    # Check every target gate before opening the first outcome/prediction file.
    manifests = {}
    for target in TARGETS:
        directory = folder / f"evaluation_{target}"
        manifest = read_json(directory / "manifest.json")
        assert manifest["status"] == "complete" and manifest["target"] == target
        assert manifest["model_records"] == 50 and manifest["baseline_records"] == 3
        assert manifest["outcome_based_selection"] is False
        assert manifest["gold_order_used"] == (target != "hotpot")
        assert manifest["models_frozen_time"] == frozen["time"]
        for name in ("manifest.json", "evaluation_lock.json"):
            relative = (directory / name).relative_to(root).as_posix()
            inputs[relative] = sha(root / relative)
        lock = read_json(directory / "evaluation_lock.json")
        assert manifest["lock"] == lock
        for relative, digest in lock.items():
            assert sha(root / relative) == digest, relative
            inputs[relative] = digest
        manifests[target] = manifest
    self_checks = self_test()
    all_records, distributions = [], {}
    for target_index, target in enumerate(TARGETS):
        directory = folder / f"evaluation_{target}"
        ids, gold, counts, masks = metadata_for(root, target, inputs)
        assert manifests[target]["n"] == len(ids)
        assert manifests[target]["strata"] == {k: int(v.sum()) for k, v in masks.items()}
        ordered = target != "hotpot"
        def load_prediction(name, prefix=""):
            path = directory / f"{name}_predictions.npz"
            inputs[path.relative_to(root).as_posix()] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                assert np.array_equal(z["ids"].astype(str), ids), path
                assert np.array_equal(z["gold_original"], gold), path
                required = [prefix + "pair"] + ([prefix + "terminal"] if ordered else [])
                prediction = {key: z[key] for key in required}
            return metric_vectors(prediction, gold, counts, ordered=ordered, prefix=prefix)
        # No aggregate results are used: every score is independently recomputed.
        primary_rows = [load_prediction(f"s{seed}_context_first100") for seed in SEEDS]
        primary = {metric: np.stack([r[metric] for r in primary_rows]) for metric in primary_rows[0]}
        comparators = {}
        for name, condition, prefix in COMPARATORS:
            if name == "frozen_e5_context":
                raw = load_prediction(condition)
                comparators[name] = {metric: np.broadcast_to(value[None], (5, len(ids)))
                                     for metric, value in raw.items()}
            else:
                rows = [load_prediction(f"s{seed}_{condition}", prefix) for seed in SEEDS]
                comparators[name] = {metric: np.stack([r[metric] for r in rows]) for metric in rows[0]}
        for stratum_index, (stratum, mask) in enumerate(masks.items()):
            keys, differences = [], []
            for comparator, _, _ in COMPARATORS:
                for metric in primary:
                    keys.append((comparator, metric))
                    differences.append((primary[metric] - comparators[comparator][metric])[:, mask])
            differences = np.stack(differences)
            values = bootstrap(differences, protocol["repetitions"], protocol["rng_seed"],
                               target_index, stratum_index, batch_size=protocol["batch_size"])
            low, high = np.percentile(values, [2.5, 97.5], axis=0, method="linear")
            for k, (comparator, metric) in enumerate(keys):
                key = "__".join((target, stratum, comparator, metric))
                distributions[key] = 100 * values[:, k]
                all_records.append({"target": target, "stratum": stratum, "metric": metric,
                    "primary_condition": "context_first100", "comparator": comparator,
                    "n_examples": int(mask.sum()), "n_seeds": 5,
                    "primary_mean_percent": 100 * float(primary[metric][:, mask].mean()),
                    "comparator_mean_percent": 100 * float(comparators[comparator][metric][:, mask].mean()),
                    "difference_pp": 100 * float(differences[k].mean()),
                    "percentile_95_ci_pp": [100 * float(low[k]), 100 * float(high[k])],
                    "seed_differences_pp": (100 * differences[k].mean(1)).tolist(),
                    "bootstrap_distribution_key": key})
            print(json.dumps({"status": "completed_stratum", "target": target, "stratum": stratum,
                              "n": int(mask.sum()), "metric_contrasts": len(keys)}), flush=True)
    assert len(all_records) == 195 and len(distributions) == 195
    assert sum(len(v) for v in STRATA.values()) == 15
    outdir.mkdir(parents=True, exist_ok=True)
    distribution_file = outdir / "bootstrap_distributions.npz"
    np.savez_compressed(distribution_file, **distributions)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex_summary(all_records), encoding="utf-8")
    report = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "analysis": "paired two-way seed-by-example percentile bootstrap",
              "repetitions": protocol["repetitions"], "rng_seed": protocol["rng_seed"],
              "seed_ids": SEEDS, "strata": STRATA, "stratum_count": 15,
              "metric_contrasts": len(all_records), "outcome_based_selection": False,
              "all_evaluations_complete_before_prediction_read": True,
              "interval_scope": "Approximate descriptive intervals; five observed training seeds; no multiplicity adjustment. Resampling examples treats examples as exchangeable within each stratum and does not model shared Wikipedia/component clusters.",
              "freshness": "MuSiQue/Hotpot/lockbox previously examined; 2Wiki remainder consists of new instances from a previously studied distribution.",
              "self_test": self_checks, "input_sha256": inputs, "records": all_records,
              "output_sha256": {distribution_file.relative_to(root).as_posix(): sha(distribution_file),
                                tex_path.relative_to(root).as_posix(): sha(tex_path)},
              "numpy_version": np.__version__, "seconds": time.time() - start}
    write_json(outdir / "all_results.json", report)
    print(json.dumps({"status": "complete", "metric_contrasts": len(all_records),
                      "seconds": report["seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tex", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.self_test:
        report = self_test()
        output = args.output or root / "results/scorers/bootstrap/self_test.json"
        write_json(output, report)
        print(json.dumps(report, indent=2))
    else:
        run(root, (args.protocol or root / "bootstrap_protocol.json").resolve(),
            (args.output or root / "results/scorers/bootstrap").resolve(),
            (args.tex or root / "results/tables/scorers_bootstrap_summary.tex").resolve())


if __name__ == "__main__":
    main()
