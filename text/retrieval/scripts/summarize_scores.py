"""Exhaustive frozen-grid summaries and LaTeX tables (NumPy only).

No winner selection, model fitting, or checkpoint choice occurs here. Default
execution requires completed lockbox, MuSiQue, Hotpot and new-2Wiki evaluations. Tables
show mean +/- sample SD; paired contrasts show mean [two-sided t95 interval].
The intervals describe five training seeds on fixed examples, not population
uncertainty. Fixed raw controls have no estimated training-seed interval.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
T95_DF4 = 2.7764451051977987
TARGETS = ("lockbox", "musique", "hotpot", "2wiki")
SEEDS = (11, 22, 33, 44, 55)
RAW_BASELINES = ("frozen_e5_direct", "frozen_e5_context", "frozen_e5_passage")
LABELS = {"direct_terminal": "Direct terminal", "context_terminal": "Context terminal",
          "context_first010": "Context + first 0.1", "context_first100": "Context + first 1",
          "context_random100": "Context + random 1", "context_teacher": "Context teacher",
          "passage_terminal": "Passage terminal", "passage_first100": "Passage + first 1",
          "question_terminal": "Question terminal", "question_first100": "Question + first 1",
          "frozen_e5_direct": "Raw E5 direct", "frozen_e5_context": "Raw E5 context",
          "frozen_e5_passage": "Raw E5 passage"}
SHORT = {"direct_terminal": "Direct", "context_terminal": "C-term", "context_first010": "C-first 0.1",
         "context_first100": "C-first 1", "context_random100": "C-random 1", "context_teacher": "C-teacher",
         "passage_terminal": "P-term", "passage_first100": "P-first 1", "question_terminal": "Q-term",
         "question_first100": "Q-first 1"}
SHORT.update(frozen_e5_direct="Raw-direct", frozen_e5_context="Raw-context", frozen_e5_passage="Raw-passage")
STRATA = {"all": "all examples", "bridge": "bridge questions", "comparison": "comparison questions",
          "clean_all_sources": "clean of all declared earlier source inputs",
          "clean_musique_train": "clean of MuSiQue training inputs",
          "clean_previously_seen_eval": "clean of previously inspected evaluation inputs",
          "clean_previous_eval": "clean of earlier evaluation inputs",
          "compositional": "compositional questions", "inference": "inference questions",
          "no_support_candidate_exposure_original_train": "without gold-support candidate exposure in original MuSiQue training"}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def statistics(values, n_examples, fixed=False):
    values = np.asarray(values, dtype=float)
    if n_examples == 0:
        check(bool(np.isnan(values).all()), "Empty stratum unexpectedly has numeric metrics")
        return {"available": False, "n_examples": 0, "reason": "empty predeclared stratum"}
    check(bool(np.isfinite(values).all()), "Nonfinite nonempty-stratum metric")
    check(len(values) == (1 if fixed else 5), "Wrong seed count for interval")
    scaled = values * 100.0
    mean = float(scaled.mean())
    if fixed:
        return {"available": True, "mean": mean, "sd": None, "half95": None,
                "low95": None, "high95": None, "num_seeds": 0,
                "deterministic_control": True, "point_evaluations": 1}
    sd = float(scaled.std(ddof=1))
    half = T95_DF4 * sd / math.sqrt(5)
    return {"available": True, "mean": mean, "sd": sd, "half95": half,
            "low95": mean - half, "high95": mean + half, "num_seeds": 5,
            "seed_values_percent_or_pp": scaled.tolist(), "seeds": list(SEEDS)}


def contrast_grid():
    grid = []
    for treated in ("context_first010", "context_first100"):
        for control in ("context_terminal", "direct_terminal", "context_teacher", "context_random100"):
            grid.append({"left": treated, "right": control,
                         "right_decoder": "distinct" if control == "direct_terminal" else "ordinary",
                         "right_source": "groups"})
        for control in RAW_BASELINES:
            grid.append({"left": treated, "right": control, "right_decoder": "ordinary",
                         "right_source": "raw_baselines"})
    grid.append({"left": "context_terminal", "right": "frozen_e5_context",
                 "right_decoder": "ordinary", "right_source": "raw_baselines"})
    for left, right in (("passage_first100", "passage_terminal"),
                        ("question_first100", "question_terminal"),
                        ("context_terminal", "passage_terminal"),
                        ("context_first100", "passage_first100"),
                        ("context_terminal", "question_terminal"),
                        ("context_first100", "question_first100"),
                        ("passage_terminal", "question_terminal"),
                        ("passage_first100", "question_first100")):
        grid.append({"left": left, "right": right, "right_decoder": "ordinary", "right_source": "groups"})
    return grid


def values_for(rows, stratum, metric):
    return [row["strata"][stratum][metric] for row in rows]


def analyze_target(root, target, protocol, frozen):
    folder = root / "results/scorers" / f"evaluation_{target}"
    manifest = read_json(folder / "manifest.json")
    check(manifest["status"] == "complete" and manifest["target"] == target,
          f"Incomplete/mismatched evaluation: {target}")
    check(manifest["outcome_based_selection"] is False, f"Unexpected outcome-based selection: {target}")
    stage = "selection" if target == "lockbox" else "refit"
    check(manifest["model_stage"] == stage and manifest["gold_order_used"] is (target != "hotpot"),
          f"Wrong checkpoint stage or gold-order use: {target}")
    lock = read_json(folder / "evaluation_lock.json")
    check(manifest["lock"] == lock and lock["results/scorers/models_frozen.json"] == sha(root / "results/scorers/models_frozen.json"),
          f"Evaluation/frozen-model source binding differs: {target}")
    check(manifest["models_frozen_time"] == frozen["time"], f"Freeze marker time differs: {target}")
    records, raw = read_json(folder / "all_results.json"), read_json(folder / "baselines.json")
    condition_names = [condition["name"] for condition in protocol["conditions"]]
    expected = {(name, seed) for name in condition_names for seed in SEEDS}
    keys = [(row["condition"], row["seed"]) for row in records]
    check(len(keys) == len(set(keys)) == 50 and set(keys) == expected, f"Missing/duplicate 10x5 grid: {target}")
    check(manifest["model_records"] == 50 and manifest["baseline_records"] == 3,
          f"Wrong completed record counts: {target}")
    raw_names = [row["condition"] for row in raw]
    check(len(raw_names) == 3 and set(raw_names) == set(RAW_BASELINES) and all(row["seed"] is None for row in raw),
          f"Missing/duplicated raw baselines: {target}")
    strata = manifest["strata"]
    expected_strata = ({"all"} if target == "lockbox" else
                      {"all", "no_support_candidate_exposure_original_train"} if target == "musique" else
                      {"all", "bridge", "comparison", "clean_all_sources", "clean_musique_train", "clean_previously_seen_eval"} if target == "hotpot" else
                      {"all", "compositional", "inference", "clean_all_sources", "clean_musique_train", "clean_previous_eval"})
    check(set(strata) == expected_strata and strata["all"] == manifest["n"], f"Wrong stratum grid: {target}")
    expected_all = {"lockbox": len(read_json(root / "results/splits/split_ids.json")["lockbox_ids"]),
                    "musique": 1252, "hotpot": 7405, "2wiki": 1563}[target]
    check(strata["all"] == expected_all, f"Wrong target example count: {target}")
    if target == "hotpot":
        check(strata["bridge"] + strata["comparison"] == strata["all"], "Hotpot type strata do not partition all examples")
    if target == "2wiki":
        check(strata["compositional"] + strata["inference"] == strata["all"], "2Wiki type strata do not partition all examples")
        check((strata["clean_musique_train"], strata["clean_previous_eval"], strata["clean_all_sources"]) == (1534, 929, 915),
              "New-2Wiki clean masks differ from the metadata lock")
    for row in records + raw:
        check(set(row["strata"]) == set(strata), f"Record stratum grid differs: {target}:{row['condition']}")
        check(row["metrics"] == row["strata"]["all"], f"All-metric alias differs: {target}")
        for stratum, n in strata.items():
            metrics = row["strata"][stratum]
            check(metrics["n"] == n, f"Metric example count differs: {target}:{stratum}")
            if target == "hotpot":
                check(not any("terminal" in key or "first" in key or "ordered" in key for key in metrics),
                      "Hotpot must not receive an invented support order")
            for key, value in metrics.items():
                if key != "n" and n:
                    check(math.isfinite(value) and 0 <= value <= 1, f"Invalid proportion: {target}:{key}")
        if row["seed"] is not None:
            name = f"results/scorers/{stage}/s{row['seed']}_{row['condition']}.pt"
            check(row["checkpoint"] == name and row["checkpoint_sha256"] == frozen["checkpoint_sha256"][name],
                  f"Wrong checkpoint or frozen digest: {target}:{name}")
    groups = {name: sorted([r for r in records if r["condition"] == name], key=lambda row: row["seed"]) for name in condition_names}
    raw_by_name = {row["condition"]: row for row in raw}
    by_stratum = {}
    for stratum, n in strata.items():
        result = {"n_examples": n, "groups": {}, "raw_baselines": {}, "paired_contrasts": [], "context_interventions": []}
        for name, rows in groups.items():
            metric_names = set(rows[0]["strata"][stratum]) - {"n"}
            check(all(set(row["strata"][stratum]) - {"n"} == metric_names for row in rows), "Seed metric schemas differ")
            result["groups"][name] = {metric: statistics(values_for(rows, stratum, metric), n) for metric in sorted(metric_names)}
        for row in raw:
            result["raw_baselines"][row["condition"]] = {
                metric: statistics([value], n, fixed=True) for metric, value in row["strata"][stratum].items() if metric != "n"}
        for contrast in contrast_grid():
            left = groups[contrast["left"]]
            reference_fixed = contrast["right_source"] == "raw_baselines"
            right = ([raw_by_name[contrast["right"]]] * len(SEEDS) if reference_fixed else groups[contrast["right"]])
            metric_pairs = {"support_em": "support_em", "support_f1": "support_f1"}
            if target != "hotpot":
                metric_pairs.update(terminal_r1="terminal_r1", first_r1="first_r1", ordered_em="ordered_em")
            if contrast["right_decoder"] == "distinct":
                metric_pairs = {"support_em": "distinct_support_em", "support_f1": "distinct_support_f1"}
                if target != "hotpot":
                    metric_pairs.update(terminal_r1="distinct_terminal_r1", ordered_em="distinct_ordered_em")
            elif reference_fixed and contrast["right"] == "frozen_e5_direct" and target != "hotpot":
                metric_pairs["ordered_em"] = "distinct_ordered_em"
            metrics = {}
            for left_metric, right_metric in metric_pairs.items():
                check(all(left_metric in a["strata"][stratum] and right_metric in b["strata"][stratum] for a, b in zip(left, right)),
                      f"Missing contrast decoder metric: {target}:{contrast}:{left_metric}")
                differences = np.asarray(values_for(left, stratum, left_metric)) - np.asarray(values_for(right, stratum, right_metric))
                metrics[left_metric] = statistics(differences, n)
            result["paired_contrasts"].append({**contrast, "reference_is_fixed": reference_fixed,
                "interval_scope": "Learned-seed variation against one fixed raw-control point" if reference_fixed else "Paired learned-seed variation",
                "metric_mapping_left_to_right": metric_pairs, "metrics": metrics})
        for name in condition_names:
            if not name.startswith("context_"):
                continue
            rows = groups[name]
            metric_pairs = {"support_em": "no_context_support_em", "support_f1": "no_context_support_f1"}
            if target != "hotpot":
                metric_pairs["terminal_r1"] = "no_context_terminal_r1"
            result["context_interventions"].append({"condition": name, "definition": "ordinary minus question-replacement inference on the same checkpoint",
                "metrics": {metric: statistics(np.asarray(values_for(rows, stratum, metric)) - np.asarray(values_for(rows, stratum, replacement)), n)
                            for metric, replacement in metric_pairs.items()}})
        by_stratum[stratum] = result
    return {"model_stage": stage, "n_examples": manifest["n"], "strata": by_stratum,
            "source_sha256": {str(path.relative_to(root)).replace("\\", "/"): sha(path) for path in
                              (folder / "all_results.json", folder / "baselines.json", folder / "manifest.json", folder / "evaluation_lock.json")}}


def analyze(root, targets):
    protocol = read_json(root / "protocol.json")
    check(protocol["optimization"]["seeds"] == list(SEEDS), "Unexpected paired-seed grid")
    check(protocol["optimization"]["residual_caps"] == [1.0, 4.0] and
          protocol["optimization"]["learning_rates"] == [0.001, 0.003], "Unexpected cap/LR pilot grid")
    check(len(protocol["conditions"]) == 10 and set(c["name"] for c in protocol["conditions"]) == set(LABELS) - set(RAW_BASELINES),
          "Unexpected condition grid")
    frozen = read_json(root / "results/scorers/models_frozen.json")
    check(frozen["status"] == "frozen" and len(frozen["checkpoint_sha256"]) == 100, "Incomplete model freeze")
    check(frozen["lock"]["protocol.json"] == sha(root / "protocol.json"), "Protocol differs from training lock")
    result = {"status": "complete", "analysis_script_sha256": sha(Path(__file__)),
              "protocol_sha256": sha(root / "protocol.json"),
              "frozen_models_sha256": sha(root / "results/scorers/models_frozen.json"),
              "condition_order": [condition["name"] for condition in protocol["conditions"]],
              "raw_baseline_order": list(RAW_BASELINES), "seeds": list(SEEDS),
              "paired_contrast_grid": contrast_grid(),
              "units": "Group metrics: percent; paired differences: percentage points; SD uses ddof=1",
              "uncertainty": "Two-sided Student-t95, df4, across five paired training seeds on fixed examples. Descriptive, no multiplicity correction or population-uncertainty claim. Raw controls are one fixed seed-zero candidate permutation and have no estimated training-seed interval.",
              "selection": "No outcome-based row, stratum, condition, checkpoint, or weight selection in this script.",
              "history_scope": {"lockbox": "Previously examined; exploratory reused source holdout",
                                "musique": "Previously inspected official development target; exploratory",
                                "hotpot": "Previously examined; exploratory external target",
                                "2wiki": "New instances fixed before the current outcomes, from an already studied dataset/distribution; not a new-dataset confirmatory test"},
              "targets": {}}
    for target in targets:
        result["targets"][target] = analyze_target(root, target, protocol, frozen)
    return result


def tex_escape(text):
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def cell(stat, difference=False):
    if not stat.get("available"):
        return "--"
    if difference:
        return f"${stat['mean']:+.2f}\\,[{stat['low95']:+.2f},{stat['high95']:+.2f}]$"
    if stat.get("deterministic_control"):
        return f"${stat['mean']:.2f}$"
    return f"${stat['mean']:.2f}\\pm{stat['sd']:.2f}$"


def table(caption, label, headers, rows):
    return [r"\begin{table}[htbp]", r"\centering", r"\caption{" + caption + "}",
            r"\label{" + label + "}", r"\small", r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{@{}l" + "c" * (len(headers) - 1) + "@{}}", r"\toprule",
            " & ".join(headers) + r"\\", r"\midrule",
            *[" & ".join(row) + r"\\" for row in rows], r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]


def write_tables(summary, directory):
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for target, target_data in summary["targets"].items():
        lines = ["% Generated by scripts/summarize_scores.py; no outcome-based row selection.",
                 "% Source JSON SHA256: " + json.dumps(target_data["source_sha256"], sort_keys=True)]
        for stratum, result in target_data["strata"].items():
            suffix = target + "-" + stratum.replace("_", "-")
            n = result["n_examples"]
            description = tex_escape(f"{target}: {STRATA[stratum]} (n={n})")
            performance_metrics = (["support_em", "support_f1"] if target == "hotpot" else
                                   ["terminal_r1", "first_r1", "support_em", "support_f1"])
            headers = (["Condition", "Set EM", "Set F1"] if target == "hotpot" else
                       ["Condition", "Terminal R@1", "First R@1", "Set EM", "Set F1"])
            rows = []
            for name in summary["condition_order"]:
                rows.append([tex_escape(LABELS[name])] + [cell(result["groups"][name][metric]) for metric in performance_metrics])
            for name in RAW_BASELINES:
                rows.append([tex_escape(LABELS[name])] + [cell(result["raw_baselines"][name][metric]) for metric in performance_metrics])
            caption = (description + r". All ten trained conditions and all three raw controls. Recall and set metrics in percent; trained rows show mean $\pm$ sample SD over five seeds. Raw rows are fixed point estimates. Ordered first/terminal labels are not assigned to Hotpot. Direct first recall, when shown, only diagnoses the terminal-trained ranking.")
            lines += table(caption, "tab:retrieval-performance-" + suffix, headers, rows)
            contrast_metrics = ["support_em", "support_f1"] if target == "hotpot" else ["terminal_r1", "support_em", "support_f1"]
            headers = (["Contrast", r"$\Delta$ set EM", r"$\Delta$ set F1"] if target == "hotpot" else
                       ["Contrast", r"$\Delta$ terminal R@1", r"$\Delta$ set EM", r"$\Delta$ set F1"])
            rows = []
            for contrast in result["paired_contrasts"]:
                control = SHORT[contrast["right"]] + (" distinct" if contrast["right_decoder"] == "distinct" else "")
                label = SHORT[contrast["left"]] + " - " + control
                rows.append([tex_escape(label)] + [cell(contrast["metrics"][metric], difference=True) for metric in contrast_metrics])
            caption = (description + r". All fixed weight/control, raw-baseline and source contrasts, in percentage points: mean difference [descriptive $95\%$ Student-$t$ seed interval]. Raw-control intervals show learned-seed variation against one fixed point. C/P/Q denote contextual-query, previous-passage, and original-question sources. Direct distinct uses raw-first plus learned-terminal pair decoding. No multiplicity adjustment.")
            lines += table(caption, "tab:retrieval-contrasts-" + suffix, headers, rows)
            rows = [[tex_escape(LABELS[row["condition"]])] + [cell(row["metrics"][metric], difference=True) for metric in contrast_metrics]
                    for row in result["context_interventions"]]
            caption = (description + r". Contextual-source intervention: ordinary inference minus replacement of every second-hop source by the original question, retaining the same trained model, first distribution and self-exclusion mask. Paired differences [descriptive $95\%$ seed interval], in pp; this intervention is distinct from the separately trained Q-source controls.")
            lines += table(caption, "tab:retrieval-interventions-" + suffix, headers, rows)
        path = directory / f"scorers_{target}_tables.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        outputs.append(path)
    combined = directory / "scorers_tables.tex"
    combined.write_text("\n".join(r"\input{" + path.stem + "}" for path in outputs) + "\n", encoding="utf-8")
    outputs.append(combined)
    if set(summary["targets"]) == set(TARGETS):
        main_rows = []
        main_grid = [("frozen_e5_context", "Raw E5 context"),
                     ("direct_terminal", "Direct + exclusion"), ("context_terminal", "Context terminal"),
                     ("context_first010", "+ First 0.1"), ("context_first100", "+ First 1"),
                     ("context_random100", "+ Random 1"), ("context_teacher", "Teacher prefix")]
        for name, label in main_grid:
            row = [tex_escape(label)]
            for target in ("musique", "hotpot", "2wiki"):
                source = summary["targets"][target]["strata"]["all"]
                group = source["raw_baselines" if name in RAW_BASELINES else "groups"][name]
                metric = "support_em" if target == "hotpot" else "terminal_r1"
                if name == "direct_terminal":
                    metric = "distinct_" + metric
                row.append(cell(group[metric]))
            main_rows.append(row)
        main_lines = table(
            r"Anchored two-hop retrieval (\%). Trained rows: full-refit five-seed mean $\pm$ SD; raw row: fixed point. Direct uses raw-first plus learned-terminal distinct decoding. MuSiQue and Hotpot were previously examined; 2Wiki contains new instances from an already studied distribution. Full grids, lockbox results and paired intervals are in the appendix.",
            "tab:retrieval-main", ["Method", "Mu R@1", "Hotpot EM", "2Wiki R@1"], main_rows)
        main_lines[4] = r"\footnotesize"
        main_lines[5] = r"\setlength{\tabcolsep}{2pt}"
        main_path = directory / "scorers_main_table.tex"
        main_path.write_text("\n".join(main_lines), encoding="utf-8")
        outputs.append(main_path)
    return outputs


def markdown(summary):
    lines = ["All scorer conditions and fixed raw controls", "", summary["uncertainty"], ""]
    for target, data in summary["targets"].items():
        for stratum, result in data["strata"].items():
            lines += [f"{target} / {stratum}: n={result['n_examples']}", ""]
            metrics = ["support_em", "support_f1"] if target == "hotpot" else ["terminal_r1", "first_r1", "support_em", "support_f1"]
            lines += ["Condition | " + " | ".join(metrics), "---|" + "|".join(["---:"] * len(metrics))]
            for key in [*summary["condition_order"], *RAW_BASELINES]:
                group = result["groups"].get(key, result["raw_baselines"].get(key))
                cells = []
                for metric in metrics:
                    value = group[metric]
                    cells.append("empty" if not value["available"] else f"{value['mean']:.2f}" if value.get("deterministic_control") else f"{value['mean']:.2f} +/- {value['sd']:.2f} SD")
                lines.append(key + " | " + " | ".join(cells))
            lines += ["", "Paired contrasts (pp; mean [95% seed interval])", ""]
            for row in result["paired_contrasts"]:
                values = []
                for metric, value in row["metrics"].items():
                    values.append(metric + ": empty" if not value["available"] else f"{metric}: {value['mean']:+.2f} [{value['low95']:+.2f}, {value['high95']:+.2f}]")
                lines.append(f"{row['left']} - {row['right']} ({row['right_decoder']}): " + "; ".join(values))
            lines += ["", "All context-source replacement effects are retained in summary.json and the LaTeX intervention tables.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--table-directory", type=Path)
    args = parser.parse_args()
    check(len(args.targets) == len(set(args.targets)), "Duplicate target requested")
    summary = analyze(args.root, args.targets)
    out = args.root / "results/scorers"
    paths = write_tables(summary, args.table_directory or args.root / "results/tables")
    summary["generated_tables"] = [str(path.relative_to(args.root)) if path.is_relative_to(args.root) else str(path) for path in paths]
    write_json(out / "summary.json", summary)
    (out / "summary.md").write_text(markdown(summary) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "targets": list(summary["targets"]),
                      "conditions_per_target": 10, "raw_baselines_per_target": 3,
                      "strata_by_target": {key: {name: value["n_examples"] for name, value in data["strata"].items()} for key, data in summary["targets"].items()},
                      "summary": str(out / "summary.json"), "tables": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
