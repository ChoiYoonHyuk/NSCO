"""Metadata-only, outcome-blind strict two-hop split for source training and evaluation.

The lockbox is evaluated only with selection-stage models. Full-data refits
contain all eligible official training rows, including this lockbox, and must
never be evaluated on it. No model, prediction, or performance file is read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

import string
import unicodedata


def normalized(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).strip()


def original_keys(row):
    result = set()
    paragraphs = {p["idx"]: p for p in row["paragraphs"]}
    for q in row["question_decomposition"]:
        result.add("id:" + str(q["id"]))
        result.add("answer:" + normalized(q["answer"]))
        p = paragraphs[q["paragraph_support_idx"]]
        result.add("paragraph:" + hashlib.sha256(
            (p["title"] + "\n" + p["paragraph_text"]).encode()).hexdigest())
    return result


def strict_keys(row):
    result = original_keys(row)
    for part in row["question_decomposition"]:
        answer = str(part["answer"]).lower().translate(str.maketrans("", "", string.punctuation))
        answer = " ".join(re.sub(r"\b(a|an|the)\b", " ", answer).split())
        result.add("answer_qa:" + answer)
    return result





ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def whitespace(text):
    return " ".join(str(text).split())


def composition_keys(row):
    result = strict_keys(row)
    result.add("composition_question:" + normalized(row["question"]))
    return result


def support_texts(row):
    paragraphs = {p["idx"]: p for p in row["paragraphs"]}
    return {hash_text(whitespace(paragraphs[q["paragraph_support_idx"]]["paragraph_text"]))
            for q in row["question_decomposition"]}


def candidate_texts(row):
    return {hash_text(whitespace(p["paragraph_text"])) for p in row["paragraphs"]}


def reasons_not_eligible(row):
    decomposition = row["question_decomposition"]
    if len(decomposition) != 2:
        return ["not_two_hop"]
    reasons = []
    references = [set(re.findall(r"#(\d+)", q["question"])) for q in decomposition]
    if references != [set(), {"1"}]:
        reasons.append("not_strict_first_then_hash1")
    supports = [q["paragraph_support_idx"] for q in decomposition]
    if len(set(supports)) != 2:
        reasons.append("not_two_distinct_support_indices")
    paragraphs = {p["idx"]: p for p in row["paragraphs"]}
    if set(supports) - set(paragraphs):
        reasons.append("invalid_support_index")
    elif len(support_texts(row)) != 2:
        reasons.append("not_two_distinct_support_texts")
    return reasons


def keyed_union(keys, positions):
    return set().union(*(keys[position] for position in positions)) if positions else set()


def locked_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"Existing metadata lock differs; use a new output directory: {path}")
        return
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--lockbox-size", type=int, default=200)
    parser.add_argument("--tune-size", type=int, default=100)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.lockbox_size, args.tune_size) <= 0:
        raise ValueError("Both held-out sizes must be positive")
    root, out = args.root, args.out or args.root / "results/splits"
    out.mkdir(parents=True, exist_ok=True)
    source = root / "data/musique_ans_v1.0_train.jsonl"
    rows = [json.loads(line) for line in source.open(encoding="utf-8")]
    source_manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 19938 and sha(source) == source_manifest["splits"]["train"]["sha256"]
    assert len({row["id"] for row in rows}) == len(rows)
    with np.load(root / "data/indices_train.npz", allow_pickle=False) as saved:
        indices = {key: saved[key] for key in ("ids", "labels", "hops", "candidate_count")}
    assert indices["ids"].tolist() == [row["id"] for row in rows]
    eligible, excluded_two_hop = [], []
    for position, row in enumerate(rows):
        decomposition = row["question_decomposition"]
        assert indices["hops"][position] == len(decomposition)
        paragraphs = row["paragraphs"]
        assert [p["idx"] for p in paragraphs] == list(range(len(paragraphs)))
        assert indices["candidate_count"][position] == len(paragraphs)
        assert 2 <= len(paragraphs) <= 20
        gold = [q["paragraph_support_idx"] for q in decomposition]
        assert indices["labels"][position, :len(gold)].tolist() == gold
        assert bool((indices["labels"][position, len(gold):] == -1).all())
        if len(decomposition) != 2:
            continue
        reason = reasons_not_eligible(row)
        if reason:
            excluded_two_hop.append({"id": row["id"], "reasons": reason})
        else:
            assert {p["idx"] for p in paragraphs if p["is_supporting"]} == set(gold)
            eligible.append(position)
    assert sum(len(row["question_decomposition"]) == 2 for row in rows) == 14376
    assert len(eligible) == 14374, f"Strict two-hop eligibility changed: {len(eligible)}"
    candidate_keys = {i: candidate_texts(rows[i]) for i in eligible}
    support_keys = {i: support_texts(rows[i]) for i in eligible}
    component_keys = {i: composition_keys(rows[i]) for i in eligible}
    lockbox = sorted(eligible, key=lambda i: hash_text("v7-lockbox:" + rows[i]["id"]))[:args.lockbox_size]
    lockbox_set, lockbox_supports = set(lockbox), keyed_union(support_keys, lockbox)
    candidates_for_tune = [i for i in eligible if i not in lockbox_set and not candidate_keys[i] & lockbox_supports]
    tune = sorted(candidates_for_tune, key=lambda i: hash_text("v7-tune:" + rows[i]["id"]))[:args.tune_size]
    assert len(lockbox) == args.lockbox_size and len(tune) == args.tune_size
    held = lockbox_set | set(tune)
    held_supports = keyed_union(support_keys, lockbox + tune)
    held_components = keyed_union(component_keys, lockbox + tune)
    remaining = [i for i in eligible if i not in held]
    candidate_purged = {i for i in remaining if candidate_keys[i] & held_supports}
    component_purged = {i for i in remaining if component_keys[i] & held_components}
    train = sorted((i for i in remaining if i not in candidate_purged | component_purged),
                   key=lambda i: hash_text("v7-train:" + rows[i]["id"]))
    assert train and not set(train) & held
    assert not keyed_union(candidate_keys, train) & held_supports
    assert not keyed_union(component_keys, train) & held_components
    assert not keyed_union(candidate_keys, tune) & lockbox_supports
    eligible_position = {original: position for position, original in enumerate(eligible)}
    full_refit = sorted(eligible, key=lambda i: hash_text("v7-full-refit:" + rows[i]["id"]))
    split = {"eligible_ids": [rows[i]["id"] for i in eligible], "eligible_indices": eligible,
             "index_coordinate_system": "*_indices address official 19938-row train; *_eligible_indices address the 14374 eligible rows in original official order",
             "lockbox_evaluation_rule": "Only selection-stage checkpoints. Full refits contain these IDs and must never be scored on this lockbox.",
             "full_refit_evaluation_rule": "Only official MuSiQue development/external data; all eligible official training examples are refit inputs."}
    for name, positions in (("train", train), ("tune", tune), ("lockbox", lockbox), ("full_refit", full_refit)):
        split[f"{name}_ids"] = [rows[i]["id"] for i in positions]
        split[f"{name}_indices"] = positions
        split[f"{name}_eligible_indices"] = [eligible_position[i] for i in positions]
        assert [split["eligible_ids"][j] for j in split[f"{name}_eligible_indices"]] == split[f"{name}_ids"]
    split["selection_ids"] = split["tune_ids"]
    split["selection_indices"] = split["tune_indices"]
    split["selection_eligible_indices"] = split["tune_eligible_indices"]
    split["purged_ids"] = [rows[i]["id"] for i in sorted(candidate_purged | component_purged)]
    locked_json(out / "split_ids.json", split)
    tune_lockbox_overlap = keyed_union(component_keys, tune) & keyed_union(component_keys, lockbox)
    manifest = {"outcome_independent_metadata_only": True, "model_outcomes_read": False,
                "source_train_sha256": sha(source), "source_indices_sha256": sha(root / "data/indices_train.npz"),
                "split_ids_sha256": sha(out / "split_ids.json"),
                "script_sha256": sha(Path(__file__)),
                "strict_rule_source_sha256": {name: sha(root / name) for name in
                    ("scripts/prepare_splits.py",)},
                "counts": {"official_train": len(rows), "two_hop": 14376, "eligible_strict_two_hop": len(eligible),
                           "excluded_two_hop": len(excluded_two_hop), "lockbox": len(lockbox),
                           "tune_candidate_pool_after_lockbox_candidate_purge": len(candidates_for_tune),
                           "tune": len(tune), "train": len(train),
                           "purged_union": len(candidate_purged | component_purged),
                           "purged_candidate_exposure": len(candidate_purged),
                           "purged_component_overlap": len(component_purged), "full_refit": len(full_refit)},
                "excluded_two_hop": excluded_two_hop,
                "rules": {"lockbox": "first SHA256(v7-lockbox:+id), no performance selection",
                          "tune": "first SHA256(v7-tune:+id) among non-lockbox rows with no lockbox gold-support text in any candidate",
                          "train": "remaining rows excluding held-out support-text candidate exposure and old strict component keys plus normalized compound questions",
                          "support_text_normalization": "body only; collapse whitespace; case/punctuation otherwise unchanged",
                          "component_keys": "old constituent IDs, NFKC/casefold/whitespace answers, QA-normalized answers, exact title+support text; add NFKC/casefold/whitespace compound question",
                          "order": "official order for eligible cache; v7-train: hash order for training; v7-full-refit: hash order for full refit"},
                "invariants": {"raw_to_cached_ids_hops_labels_candidate_counts": "passed",
                               "eligible_two_distinct_supports_and_strict_hash1": "passed",
                               "train_vs_heldout_component_overlap": 0,
                               "heldout_gold_support_text_in_train_candidates": 0,
                               "lockbox_gold_support_text_in_tune_candidates": 0},
                "tune_vs_lockbox_component_key_overlap": {"count": len(tune_lockbox_overlap),
                    "by_key_type": dict(Counter(key.split(":", 1)[0] for key in tune_lockbox_overlap)),
                    "note": "Reported without altering declared tune selection; only support-text candidate purge was specified between tune and lockbox."},
                "lockbox_scope": "New supervised-head selection-stage evaluation only; pretrained encoders/earlier aggregate studies prevent an untouched-data or contamination-free claim. Never evaluate full refits on this lockbox."}
    locked_json(out / "split_manifest.json", manifest)
    print(json.dumps({"counts": manifest["counts"], "invariants": manifest["invariants"],
                      "tune_vs_lockbox_component_key_overlap": manifest["tune_vs_lockbox_component_key_overlap"],
                      "source_train_sha256": manifest["source_train_sha256"],
                      "split_ids_sha256": manifest["split_ids_sha256"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
