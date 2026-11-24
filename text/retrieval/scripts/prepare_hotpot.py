"""Prepare and lock official HotpotQA distractor dev without model outcomes.

New output folder only. Original candidate order is preserved; support labels
are unordered. Answers/sentence labels are stored separately from model text.
Only public raw metadata is read, never model predictions or result metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import string
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URLS = ["https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"]
HF_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
HF_PARQUET_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
HF_URL = f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{HF_REVISION}/distractor/validation-00000-of-00001.parquet?download=true"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def lock_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"Existing outcome-independent ID lock differs: {path}")
    else:
        write_json(path, value)


def normalized_question(value):
    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(?:a|an|the)\b", " ", value).split())


def whitespace(value):
    return " ".join(value.split())


def download(out):
    path = out / "hotpot_dev_distractor_v1.json"
    provenance = out / "download_provenance.json"
    if path.exists():
        info = json.loads(provenance.read_text(encoding="utf-8"))
        assert info["raw_sha256"] == sha(path), "Existing raw file changed"
        return path, info
    errors = []
    for url in URLS:
        temporary = path.with_name(path.name + ".partial")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ICML-retrieval-research/1.0"})
            with urllib.request.urlopen(request, timeout=15) as response, temporary.open("wb") as stream:
                resolved_url = response.geturl()
                content_type = response.headers.get("Content-Type")
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
            # Parse before committing; never silently save an HTML error page.
            rows = json.loads(temporary.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or len(rows) != 7405:
                raise ValueError("Official distractor dev must contain 7,405 rows")
            temporary.replace(path)
            info = {"requested_url": url, "resolved_url": resolved_url,
                    "content_type": content_type, "raw_sha256": sha(path),
                    "raw_bytes": path.stat().st_size,
                    "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                    "failed_earlier_urls": errors}
            write_json(provenance, info)
            return path, info
        except Exception as exc:
            errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    # The author-linked host can be unavailable. The approved archival fallback
    # is pinned to the HotpotQA HF namespace's documented Parquet conversion.
    # Never claim its reconstructed JSON is byte-identical to the author file.
    print(json.dumps({"official_host_failed": errors, "fallback": HF_URL}), flush=True)
    import pyarrow.parquet as pq
    archive = out / "source_hf_validation.parquet"
    if not archive.exists():
        temporary = archive.with_name(archive.name + ".partial")
        request = urllib.request.Request(HF_URL, headers={"User-Agent": "ICML-retrieval-research/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block: break
                stream.write(block)
        assert sha(temporary) == HF_PARQUET_SHA256, "Pinned HotpotQA archive hash mismatch"
        temporary.replace(archive)
    assert sha(archive) == HF_PARQUET_SHA256
    stored = pq.read_table(archive).to_pylist()
    assert len(stored) == 7405 and len({x["id"] for x in stored}) == 7405
    reconstructed = []
    for row in stored:
        context, facts = row["context"], row["supporting_facts"]
        if isinstance(context, dict):
            assert len(context["title"]) == len(context["sentences"])
            context = [[t, ss] for t, ss in zip(context["title"], context["sentences"])]
        else:
            context = [[p["title"], p["sentences"]] for p in context]
        if isinstance(facts, dict):
            assert len(facts["title"]) == len(facts["sent_id"])
            facts = [[t, i] for t, i in zip(facts["title"], facts["sent_id"])]
        else:
            facts = [[f["title"], f["sent_id"]] for f in facts]
        reconstructed.append({"_id": row["id"], "question": row["question"], "answer": row["answer"],
                              "type": row["type"], "level": row["level"],
                              "context": context, "supporting_facts": facts})
    write_json(path, reconstructed)
    info = {"requested_url": HF_URL, "source_kind": "Pinned HotpotQA-namespace HF archival Parquet conversion",
            "official_urls": URLS, "official_host_failures": errors,
            "mirror_repository": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
            "mirror_revision": HF_REVISION, "raw_source_file": archive.name,
            "raw_source_sha256": sha(archive), "raw_source_bytes": archive.stat().st_size,
            "published_expected_parquet_sha256": HF_PARQUET_SHA256,
            "conversion_provenance": f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/commit/{HF_REVISION}",
            "loader_provenance": "https://huggingface.co/datasets/hotpotqa/hotpot_qa/blob/087b2e4/hotpot_qa.py",
            "conversion": "Read pinned distractor validation Parquet only; rename id to _id and invert list-of-struct/struct-of-list representation without sorting records, contexts, sentences, or annotations",
            "raw_sha256": sha(path), "raw_bytes": path.stat().st_size,
            "reconstructed_json_sha256": sha(path), "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "identity_limit": "7,405 unique dev IDs, fields, and nested sequence order verified against pinned archive. Original author-host JSON byte identity cannot be independently checked while that host is unavailable."}
    write_json(provenance, info)
    return path, info


def convert(row):
    if not isinstance(row, dict) or not isinstance(row.get("_id"), str) or not row["_id"]:
        return None, None, "invalid_id"
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        return None, None, "invalid_question"
    if not isinstance(row.get("type"), str) or not row["type"]:
        return None, None, "invalid_type"
    context = row.get("context")
    if not isinstance(context, list) or not 2 <= len(context) <= 20:
        return None, None, "candidate_count_outside_2_to_20"
    if any(not isinstance(p, list) or len(p) != 2 or not isinstance(p[0], str) or not p[0]
           or not isinstance(p[1], list) or not p[1]
           or not all(isinstance(s, str) for s in p[1]) for p in context):
        return None, None, "invalid_context_schema"
    titles = [p[0] for p in context]
    if len(titles) != len(set(titles)):
        return None, None, "duplicate_candidate_title"
    facts = row.get("supporting_facts")
    if not isinstance(facts, list) or not facts or any(
        not isinstance(f, list) or len(f) != 2 or not isinstance(f[0], str)
        or type(f[1]) is not int for f in facts
    ):
        return None, None, "invalid_supporting_fact_schema"
    gold_titles = {f[0] for f in facts}
    if len(gold_titles) != 2:
        return None, None, "not_exactly_two_support_titles"
    if not gold_titles <= set(titles):
        return None, None, "support_title_missing_from_context"
    # Paragraph labels remain usable even if the sentence annotation has an
    # out-of-range index. Preserve such rows and flag the annotation separately.
    invalid_sentences = [f for f in facts if not 0 <= f[1] < len(context[titles.index(f[0])][1])]
    example = {"id": row["_id"], "question": row["question"],
               "candidates": [{"title": t, "text": " ".join(ss)} for t, ss in context],
               "gold_support_indices": sorted(titles.index(t) for t in gold_titles),
               "type": row["type"]}
    annotations = {"id": row["_id"], "answer": row.get("answer"),
                   "supporting_facts": facts, "level": row.get("level"),
                   "invalid_support_sentence_indices": invalid_sentences,
                   "gold_support_titles_unordered": sorted(gold_titles)}
    assert set(example) == {"id", "question", "candidates", "gold_support_indices", "type"}
    assert len(example["gold_support_indices"]) == 2
    return example, annotations, "eligible"


def empty_reference():
    return {key: set() for key in ["questions", "component_questions", "titles", "texts", "ws_texts", "pairs", "ws_pairs"]}


def add_candidate(reference, title, text):
    reference["titles"].add(title)
    reference["texts"].add(text)
    reference["ws_texts"].add(whitespace(text))
    reference["pairs"].add((title, text))
    reference["ws_pairs"].add((title, whitespace(text)))


def read_reference(path, kind):
    reference, count, paragraphs = empty_reference(), 0, 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            reference["questions"].add(normalized_question(row["question"]))
            if kind == "musique":
                for step in row["question_decomposition"]:
                    reference["component_questions"].add(normalized_question(step["question"]))
            for paragraph in row["paragraphs"]:
                add_candidate(reference, paragraph["title"], paragraph["paragraph_text"])
                paragraphs += 1
    return reference, {"path": str(path), "sha256": sha(path), "rows": count,
                       "actual_candidate_paragraph_occurrences": paragraphs,
                       "unique_reference_counts": {key: len(value) for key, value in reference.items()},
                       "scope": "All actual candidate paragraphs, including distractors; composed and available constituent question strings"}


def overlap_flags(example, reference):
    q = normalized_question(example["question"])
    supports = [example["candidates"][j] for j in example["gold_support_indices"]]
    candidates = example["candidates"]
    flags = {
        "normalized_question": q in reference["questions"],
        "normalized_constituent_question": q in reference["component_questions"],
        "support_title_exact": any(p["title"] in reference["titles"] for p in supports),
        "support_text_exact_raw": any(p["text"] in reference["texts"] for p in supports),
        "support_text_exact_whitespace_normalized": any(whitespace(p["text"]) in reference["ws_texts"] for p in supports),
        "support_title_and_text_exact_raw": any((p["title"], p["text"]) in reference["pairs"] for p in supports),
        "support_title_and_text_exact_whitespace_normalized": any((p["title"], whitespace(p["text"])) in reference["ws_pairs"] for p in supports),
        "any_candidate_text_exact_raw": any(p["text"] in reference["texts"] for p in candidates),
        "any_candidate_text_exact_whitespace_normalized": any(whitespace(p["text"]) in reference["ws_texts"] for p in candidates),
    }
    flags["clean_question_and_support_text"] = not any(flags[k] for k in [
        "normalized_question", "normalized_constituent_question",
        "support_text_exact_raw", "support_text_exact_whitespace_normalized"])
    return flags


def jsonl(path, rows):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def self_test():
    source = {"_id": "s", "question": "Q?", "type": "comparison", "answer": "SECRET",
              "context": [["B", ["b1", "b2"]], ["D", ["d"]], ["A", ["a"]]],
              "supporting_facts": [["A", 0], ["B", 1]]}
    x, annotation, reason = convert(source)
    assert reason == "eligible" and x["gold_support_indices"] == [0, 2]
    assert [p["title"] for p in x["candidates"]] == ["B", "D", "A"]
    assert x["candidates"][0]["text"] == "b1 b2" and "answer" not in x
    assert annotation["answer"] == "SECRET"
    reversed_source = dict(source, supporting_facts=list(reversed(source["supporting_facts"])))
    assert convert(reversed_source)[0] == x
    reference = empty_reference(); add_candidate(reference, "B", "b1\n b2")
    flags = overlap_flags(x, reference)
    assert not flags["support_text_exact_raw"] and flags["support_text_exact_whitespace_normalized"]
    assert not flags["clean_question_and_support_text"]
    source["supporting_facts"][0][1] = 999
    assert convert(source)[1]["invalid_support_sentence_indices"] == [["A", 999]]
    print("Self-test passed: unordered gold invariance, original candidate order, annotation separation, whitespace overlap, sentence-warning preservation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    root = args.root.resolve(); out = root / "data/hotpot"
    out.mkdir(parents=True, exist_ok=True)
    raw_path, download_info = download(out)
    rows = json.loads(raw_path.read_text(encoding="utf-8"))
    assert len(rows) == 7405 and len({x["_id"] for x in rows}) == 7405
    examples, annotations, exceptions, excluded, warnings = [], [], Counter(), [], Counter()
    for row in rows:
        example, annotation, reason = convert(row)
        exceptions[reason] += 1
        if example is None:
            excluded.append({"id": row.get("_id"), "reason": reason}); continue
        examples.append(example); annotations.append(annotation)
        if annotation["invalid_support_sentence_indices"]:
            warnings["invalid_support_sentence_index_examples_retained_for_paragraph_metrics"] += 1
        if len(example["candidates"]) != 10:
            warnings["candidate_count_not_ten"] += 1
        if example["type"] not in {"bridge", "comparison"}:
            warnings["unexpected_question_type_retained"] += 1
        # Independent projection check against the original source record.
        assert example["question"] == row["question"] and example["id"] == row["_id"]
        assert all(p["title"] == original[0] and p["text"] == " ".join(original[1])
                   for p, original in zip(example["candidates"], row["context"]))
        assert {example["candidates"][j]["title"] for j in example["gold_support_indices"]} == {f[0] for f in row["supporting_facts"]}
    assert examples
    ids = [x["id"] for x in examples]
    lock_json(out / "selection_ids.json", ids)
    sources = {"musique_train": (root / "data/musique_ans_v1.0_train.jsonl", "musique"),
               "previously_seen_musique_dev": (root / "data/musique_ans_v1.0_dev.jsonl", "musique"),
               "previously_seen_2wiki_selected": (root / "data/wiki_reference/examples.jsonl", "2wiki")}
    flags_by_source, source_reports = {}, {}
    for name, (path, kind) in sources.items():
        reference, source_info = read_reference(path, kind)
        if name == "musique_train": assert source_info["rows"] == 19938
        if name == "previously_seen_musique_dev": assert source_info["rows"] == 2417
        if name == "previously_seen_2wiki_selected": assert source_info["rows"] == 2000
        flags_by_source[name] = {x["id"]: overlap_flags(x, reference) for x in examples}
        counts = {key: sum(f[key] for f in flags_by_source[name].values())
                  for key in next(iter(flags_by_source[name].values()))}
        source_reports[name] = {**source_info, "example_overlap_counts": counts}
        del reference
    by_id = {i: {name: flags_by_source[name][i] for name in sources} for i in ids}
    clean_train = [i for i in ids if by_id[i]["musique_train"]["clean_question_and_support_text"]]
    clean_seen = [i for i in ids if all(by_id[i][n]["clean_question_and_support_text"] for n in [
        "previously_seen_musique_dev", "previously_seen_2wiki_selected"])]
    clean_all = [i for i in ids if all(by_id[i][n]["clean_question_and_support_text"] for n in sources)]
    subsets = {"clean_musique_train_selection_ids.json": clean_train,
               "clean_previously_seen_eval_selection_ids.json": clean_seen,
               "clean_all_sources_selection_ids.json": clean_all}
    for name, selected in subsets.items(): lock_json(out / name, selected)
    jsonl(out / "examples.jsonl", examples)
    jsonl(out / "annotations.jsonl", annotations)
    write_json(out / "overlap_by_id.json", by_id)
    write_json(out / "eligibility_exceptions.json", excluded)
    overlap = {"method": "Metadata only, before any Hotpot model outcome; separate source sets, no prediction/result files read",
               "normalization": "Questions: lowercase, ASCII punctuation removal, English article removal, whitespace collapse. Paragraphs: exact raw or whitespace-only normalized string equality. Titles: exact.",
               "sources": source_reports,
               "clean_rule": "No normalized composed/constituent question match and no gold support paragraph text match against ANY actual candidate paragraph of the specified reference source(s). Titles and overlap confined to Hotpot distractors do not trigger exclusion.",
               "clean_subset_counts": {name: len(value) for name, value in subsets.items()},
               "limitations": "Exact metadata filtering cannot eliminate paraphrases, shared Wikipedia content or pretraining exposure; previously examined evaluation distributions already informed method design."}
    write_json(out / "metadata_overlap.json", overlap)
    files = ["selection_ids.json", "examples.jsonl", "annotations.jsonl", "overlap_by_id.json",
             "metadata_overlap.json", "eligibility_exceptions.json", *subsets]
    manifest = {"study": "New outcome-uninspected HotpotQA distractor-dev paragraph support-set transfer diagnostic",
                "status": "Preparation, schema audit, overlap audit and ID locks complete; no model encoding/inference/outcomes",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": download_info, "source_rows": len(rows), "eligible_rows": len(examples),
                "eligibility_first_failure_counts": dict(exceptions), "annotation_warnings": dict(warnings),
                "eligible_types": dict(Counter(x["type"] for x in examples)),
                "candidate_counts": dict(Counter(len(x["candidates"]) for x in examples)),
                "selection_rule": "All schema-eligible official distractor dev examples in original source order; no type or outcome-based sampling",
                "candidate_order": "Original official context order preserved exactly",
                "support_policy": "Exactly two unique gold titles, mapped to unique candidate indices; stored sorted by candidate index solely as an unordered set. No first/terminal or directed path label is derived.",
                "sentence_policy": "Sentence annotations kept in annotations.jsonl. Invalid sentence indices are flagged, not used for paragraph metrics or encoding, and do not invalidate an unambiguous paragraph-level title label.",
                "encoder_input_policy": "Only question and candidates[].title/text may enter encoder. answer/supporting_facts remain in separate annotations.jsonl; gold_support_indices and type are scoring/metadata only.",
                "evaluation_boundary": "No Hotpot training examples, models, predictions, metrics, hyperparameter choice, or checkpoint selection are read or computed by this script. Future external inference only after model freeze authorization.",
                "recommended_metrics": ["paragraph support-set EM@2", "paragraph support-set F1@2"],
                "metric_boundary": "Two distinct predicted paragraphs; neither ordered-support nor sentence-level official Hotpot supporting-fact/answer/joint QA performance",
                "primary_links": {"homepage": "https://hotpotqa.github.io/",
                                  "paper": "https://aclanthology.org/D18-1259/",
                                  "official_download_script": "https://github.com/hotpotqa/hotpot/blob/master/download.sh",
                                  "official_schema_code": "https://github.com/hotpotqa/hotpot/blob/master/prepro.py",
                                  "official_set_scorer": "https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py",
                                  "license_statement": "https://github.com/hotpotqa/hotpot#license"},
                "license": {"dataset": "CC BY-SA 4.0", "official_baseline_code": "Apache 2.0",
                            "dataset_license_url": "https://creativecommons.org/licenses/by-sa/4.0/"},
                "metadata_overlap": overlap,
                "primary_clean_selection_file": "clean_all_sources_selection_ids.json",
                "files_sha256": {name: sha(out / name) for name in files},
                "script_sha256": sha(Path(__file__))}
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
