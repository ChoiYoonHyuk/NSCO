"""Lock an outcome-uninspected, strictly ordered 2Wiki official-dev subset.

Downloads the April 7 official archive, reads/extracts only dev.json, and derives
order from a unique two-edge directed evidence path, never serialized list order.
No model is loaded and no model outcome is calculated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import string
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"
OFFICIAL_REPOSITORY = "https://github.com/Alab-NII/2wikimultihop"
PAPER = "https://aclanthology.org/2020.coling-main.580/"
LIMIT = 2000


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_write(path, obj):
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def source_dev(out):
    archive = out / "official_data_ids_april7.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.partial")
        request = urllib.request.Request(URL, headers={"User-Agent": "ICML-retrieval-research/1.0"})
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as target:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Official download is not a ZIP archive; refusing to parse an HTML response")
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        dev = [member for member in members if not member.is_dir()
               and PurePosixPath(member.filename.replace("\\", "/")).name == "dev.json"
               and "__MACOSX" not in member.filename]
        if len(dev) != 1:
            raise ValueError(f"Expected exactly one dev.json member, found {[m.filename for m in dev]}")
        member = dev[0]
        if member.file_size > 1024 ** 3:
            raise ValueError("Unexpected dev member larger than 1 GiB")
        # Never use extract/extractall: archive path names cannot affect destination.
        destination = out / "official_dev.json"
        assert destination.resolve().parent == out.resolve()
        temporary = destination.with_suffix(".json.partial")
        with source.open(member) as stream, temporary.open("wb") as target:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
        temporary.replace(destination)
        archive_manifest = {
            "url": URL, "archive_sha256": digest(archive),
            "archive_bytes": archive.stat().st_size,
            "extracted_member": member.filename, "dev_sha256": digest(destination),
            "dev_bytes": destination.stat().st_size,
            "archive_members": [{"name": m.filename, "bytes": m.file_size} for m in members],
            "extraction_policy": "Only official dev.json copied to a fixed safe destination. Train/test members are not extracted or parsed.",
        }
    rows = json.loads(destination.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Official dev must be a JSON list")
    return rows, archive_manifest


def derive(row):
    """Return (example, reason), with one mutually exclusive first-failure reason."""
    if row.get("type") not in {"inference", "compositional"}:
        return None, "excluded_question_type"
    if not isinstance(row.get("_id"), str) or not row["_id"]:
        return None, "invalid_id"
    if not isinstance(row.get("question"), str) or not row["question"]:
        return None, "invalid_question"
    context = row.get("context")
    if not isinstance(context, list) or not 2 <= len(context) <= 20:
        return None, "candidate_count_outside_2_to_20"
    if any(not isinstance(p, (list, tuple)) or len(p) != 2 or not isinstance(p[0], str)
           or not p[0] or not isinstance(p[1], list) or not p[1]
           or not all(isinstance(s, str) for s in p[1]) for p in context):
        return None, "invalid_context_schema"
    titles = [p[0] for p in context]
    if len(set(titles)) != len(titles):
        return None, "duplicate_candidate_title"
    facts = row.get("supporting_facts")
    if not isinstance(facts, list) or not facts:
        return None, "invalid_supporting_facts"
    if any(not isinstance(f, (list, tuple)) or len(f) != 2 or not isinstance(f[0], str)
           or not isinstance(f[1], int) or isinstance(f[1], bool) for f in facts):
        return None, "invalid_supporting_facts"
    gold_titles = set(f[0] for f in facts)
    if len(gold_titles) != 2:
        return None, "not_exactly_two_gold_titles"
    if not gold_titles.issubset(titles):
        return None, "gold_title_missing_from_context"
    if any(not 0 <= f[1] < len(context[titles.index(f[0])][1]) for f in facts):
        return None, "invalid_gold_sentence_index"
    edges = row.get("evidences")
    if not isinstance(edges, list) or len(edges) != 2:
        return None, "not_exactly_two_evidence_edges"
    if any(not isinstance(e, (list, tuple)) or len(e) != 3
           or not all(isinstance(x, str) and x for x in e) for e in edges):
        return None, "invalid_evidence_schema"
    orders = [(i, j) for i, j in ((0, 1), (1, 0))
              if edges[i][2] == edges[j][0]
              and len({edges[i][0], edges[i][2], edges[j][2]}) == 3]
    if len(orders) != 1:
        return None, "no_unique_directed_two_edge_path"
    first, second = orders[0]
    subjects = [edges[first][0], edges[second][0]]
    if set(subjects) != gold_titles:
        return None, "evidence_subjects_not_exact_gold_titles"
    labels = [titles.index(subject) for subject in subjects]
    result = {
        "id": row["_id"], "question": row["question"], "type": row["type"],
        "paragraphs": [{"idx": i, "title": title, "paragraph_text": " ".join(sentences)}
                       for i, (title, sentences) in enumerate(context)],
        "support_labels": labels, "hops": 2, "source_split": "official_dev",
        "order_derivation": {
            "method": "Unique directed two-edge evidence path; exact subject-to-gold-title matches; never list order",
            "evidence_input_positions": [first, second],
            "ordered_evidence": [list(edges[first]), list(edges[second])],
            "path_entities": [subjects[0], subjects[1], edges[second][2]],
            "ordered_support_titles": subjects,
            "answer_exactly_matches_path_tail": row.get("answer") == edges[second][2],
        },
    }
    return result, "eligible"


def independent_audit(example, source):
    """Degree-based path audit, independent of derive's two-permutation test."""
    edges = source["evidences"]
    outgoing = Counter(e[0] for e in edges)
    incoming = Counter(e[2] for e in edges)
    nodes = set(outgoing) | set(incoming)
    assert len(nodes) == 3
    roots = [n for n in nodes if incoming[n] == 0 and outgoing[n] == 1]
    middle = [n for n in nodes if incoming[n] == 1 and outgoing[n] == 1]
    tails = [n for n in nodes if incoming[n] == 1 and outgoing[n] == 0]
    assert len(roots) == len(middle) == len(tails) == 1
    successor = {e[0]: e[2] for e in edges}
    assert successor[roots[0]] == middle[0] and successor[middle[0]] == tails[0]
    titles = [p[0] for p in source["context"]]
    expected = [titles.index(roots[0]), titles.index(middle[0])]
    assert expected == example["support_labels"]
    assert {titles[i] for i in expected} == {fact[0] for fact in source["supporting_facts"]}
    assert example["question"] == source["question"] and example["id"] == source["_id"]
    assert example["hops"] == 2 and len(set(expected)) == 2
    for paragraph, original in zip(example["paragraphs"], source["context"]):
        assert paragraph["title"] == original[0] and paragraph["paragraph_text"] == " ".join(original[1])


def qa_normalize(text):
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def whitespace_normalize(text):
    return " ".join(text.split())


def metadata_overlap(root, examples, sources):
    """Metadata-only overlap against the complete official MuSiQue training file."""
    source = root / "data/musique_ans_v1.0_train.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"Required pre-evaluation metadata overlap source missing: {source}")
    questions, component_questions, answers, final_answers = set(), set(), set(), set()
    texts, normalized_texts, pairs, normalized_pairs, titles = set(), set(), set(), set(), set()
    support_only_texts, support_only_normalized_texts = set(), set()
    rows = 0
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            questions.add(qa_normalize(row["question"]))
            final_answers.add(qa_normalize(row["answer"]))
            final_answers.update(qa_normalize(x) for x in row.get("answer_aliases", []))
            answers.add(qa_normalize(row["answer"]))
            answers.update(qa_normalize(x) for x in row.get("answer_aliases", []))
            for step in row["question_decomposition"]:
                component_questions.add(qa_normalize(step["question"]))
                answers.add(qa_normalize(step["answer"]))
            for paragraph in row["paragraphs"]:
                title, text = paragraph["title"], paragraph["paragraph_text"]
                titles.add(title)
                texts.add(text)
                normalized_texts.add(whitespace_normalize(text))
                pairs.add((title, text))
                normalized_pairs.add((title, whitespace_normalize(text)))
                if paragraph["is_supporting"]:
                    support_only_texts.add(text)
                    support_only_normalized_texts.add(whitespace_normalize(text))
    answers.discard("")
    final_answers.discard("")
    by_id = {}
    for example in examples:
        question = qa_normalize(example["question"])
        supports = [example["paragraphs"][label] for label in example["support_labels"]]
        final = qa_normalize(sources[example["id"]].get("answer", ""))
        endpoints = [qa_normalize(x) for x in example["order_derivation"]["path_entities"]]
        flags = {
            "composed_question_qa_normalized": question in questions,
            "component_question_qa_normalized": question in component_questions,
            "support_title_exact": any(p["title"] in titles for p in supports),
            "support_text_exact_raw": any(p["paragraph_text"] in texts for p in supports),
            "support_text_exact_whitespace_normalized": any(whitespace_normalize(p["paragraph_text"]) in normalized_texts for p in supports),
            "support_title_and_text_exact_raw": any((p["title"], p["paragraph_text"]) in pairs for p in supports),
            "support_title_and_text_exact_whitespace_normalized": any((p["title"], whitespace_normalize(p["paragraph_text"])) in normalized_pairs for p in supports),
            "support_text_vs_musique_support_only_exact_raw": any(p["paragraph_text"] in support_only_texts for p in supports),
            "support_text_vs_musique_support_only_whitespace_normalized": any(whitespace_normalize(p["paragraph_text"]) in support_only_normalized_texts for p in supports),
            "final_answer_vs_final_answer_qa_normalized": bool(final) and final in final_answers,
            "final_answer_vs_any_answer_qa_normalized": bool(final) and final in answers,
            "any_path_entity_vs_any_answer_qa_normalized": any(x and x in answers for x in endpoints),
            "path_root_vs_any_answer_qa_normalized": bool(endpoints[0]) and endpoints[0] in answers,
            "path_bridge_vs_any_answer_qa_normalized": bool(endpoints[1]) and endpoints[1] in answers,
            "path_tail_vs_any_answer_qa_normalized": bool(endpoints[2]) and endpoints[2] in answers,
        }
        flags["non_overlap_question_and_support_text"] = not any(flags[key] for key in (
            "composed_question_qa_normalized", "component_question_qa_normalized",
            "support_text_exact_raw", "support_text_exact_whitespace_normalized"))
        by_id[example["id"]] = flags
    counts = {key: sum(flags[key] for flags in by_id.values()) for key in next(iter(by_id.values()), {})}
    report = {
        "status": "Metadata only; computed before any external model outcome",
        "comparison_source": str(source), "source_sha256": digest(source), "source_rows": rows,
        "comparison_scope": "External GOLD supporting paragraphs compared against ALL actual candidate paragraphs in complete official MuSiQue train, including distractors and internal-selection holdouts. Support-only reference matches are additionally reported.",
        "normalization": "Questions/answers: SQuAD-style lowercase, ASCII punctuation removal, English article removal, whitespace collapse. Paragraph exact raw and whitespace-only-normalized matches are reported separately; title comparisons are exact.",
        "unique_reference_counts": {"composed_questions": len(questions), "component_questions": len(component_questions),
                                    "all_candidate_texts_raw": len(texts), "all_candidate_texts_whitespace_normalized": len(normalized_texts),
                                    "all_candidate_titles": len(titles), "support_only_texts_raw": len(support_only_texts),
                                    "support_only_texts_whitespace_normalized": len(support_only_normalized_texts),
                                    "all_answers_qa_normalized": len(answers),
                                    "final_answers_qa_normalized": len(final_answers)},
        "eligible_example_overlap_counts": counts,
        "non_overlap_rule": "No QA-normalized composed/component question match and no external gold support text matching any official MuSiQue training candidate paragraph (raw or whitespace-normalized exact match). Shared titles and answer strings are reported but do not trigger exclusion.",
        "limitations": "Exact metadata matches cannot establish contamination freedom. Wikipedia content, paraphrases, and E5 pretraining exposure to MuSiQue seed QA datasets may still overlap.",
    }
    return by_id, report


def self_test():
    row = {"_id": "example", "type": "inference", "question": "question",
           "context": [["A", ["A mentions B."]], ["B", ["B mentions C."]]],
           "supporting_facts": [["B", 0], ["A", 0]],
           "evidences": [["B", "r2", "C"], ["A", "r1", "B"]], "answer": "C"}
    example, reason = derive(row)
    assert reason == "eligible" and example["support_labels"] == [0, 1]
    independent_audit(example, row)
    row["evidences"].reverse()
    reordered, reason = derive(row)
    assert reason == "eligible" and reordered["support_labels"] == [0, 1]
    row["evidences"] = [["A", "r1", "B"], ["B", "r2", "A"]]
    assert derive(row)[1] == "no_unique_directed_two_edge_path"
    row["evidences"] = [["a", "r1", "B"], ["B", "r2", "C"]]
    assert derive(row)[1] == "evidence_subjects_not_exact_gold_titles"
    print("Self-test passed: unordered facts/edges, cycle exclusion, and exact-title exclusion.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    out = args.root.resolve() / "data/wiki_reference"
    out.mkdir(parents=True, exist_ok=True)
    rows, archive = source_dev(out)
    id_counts = Counter(row.get("_id") for row in rows if isinstance(row.get("_id"), str))
    reasons, types = Counter(), Counter(row.get("type", "MISSING") for row in rows)
    eligible, sources = [], {}
    for row in rows:
        if isinstance(row.get("_id"), str) and id_counts[row["_id"]] > 1:
            reasons["duplicate_source_id"] += 1
            continue
        example, reason = derive(row)
        reasons[reason] += 1
        if example is not None:
            independent_audit(example, row)
            eligible.append(example)
            sources[example["id"]] = row
    eligible.sort(key=lambda row: (hashlib.sha256(row["id"].encode()).hexdigest(), row["id"]))
    selected = eligible[:LIMIT]
    if not selected:
        raise ValueError(f"No unambiguous eligible examples: {dict(reasons)}")
    overlap_by_id, overlap_report = metadata_overlap(args.root.resolve(), eligible, sources)
    ids = [row["id"] for row in selected]
    lock = out / "selection_ids.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != ids:
        raise RuntimeError("Existing external selection lock differs; refusing to change evaluation examples")
    json_write(lock, ids)
    non_overlap_ids = [identity for identity in ids
                       if overlap_by_id[identity]["non_overlap_question_and_support_text"]]
    # A named stronger audit preserves any earlier support-only subset lock.
    non_overlap_lock = out / "non_overlap_all_candidates_selection_ids.json"
    if non_overlap_lock.exists() and json.loads(non_overlap_lock.read_text(encoding="utf-8")) != non_overlap_ids:
        raise RuntimeError("Existing non-overlap external lock differs; refusing to change subset")
    json_write(non_overlap_lock, non_overlap_ids)
    json_write(out / "non_overlap_all_candidates_eligible_ids.json", [row["id"] for row in eligible
               if overlap_by_id[row["id"]]["non_overlap_question_and_support_text"]])
    json_write(out / "metadata_overlap_all_candidates_by_id.json", overlap_by_id)
    overlap_report["selected_example_overlap_counts"] = {
        key: sum(overlap_by_id[identity][key] for identity in ids)
        for key in next(iter(overlap_by_id.values()), {})}
    overlap_report["non_overlap_selected_count"] = len(non_overlap_ids)
    overlap_report["non_overlap_selected_at_least_200"] = len(non_overlap_ids) >= 200
    overlap_report["non_overlap_subset_scope"] = "Subset of the primary hash-first-2000 selected examples, fixed before predictions; no additional examples are selected using model outcomes."
    json_write(out / "metadata_overlap_all_candidates.json", overlap_report)
    examples_path = out / "examples.jsonl"
    temporary = examples_path.with_suffix(".jsonl.partial")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in selected:
            independent_audit(row, sources[row["id"]])
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(examples_path)
    json_write(out / "eligible_ids.json", [row["id"] for row in eligible])
    manifest = {
        "study": "External transfer only: strictly derived two-hop 2Wiki official-dev subset",
        "status": "Selection and structural audit only; no model predictions or outcomes computed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_repository": OFFICIAL_REPOSITORY, "paper": PAPER, "source": archive,
        "source_rows": len(rows), "source_types": dict(types),
        "source_key_histogram": dict(Counter(key for row in rows for key in row)),
        "first_failure_counts": dict(reasons), "eligible_count": len(eligible),
        "eligible_types": dict(Counter(row["type"] for row in eligible)),
        "eligible_answer_tail_exact_match_count": sum(row["order_derivation"]["answer_exactly_matches_path_tail"] for row in eligible),
        "selected_count": len(selected), "selected_types": dict(Counter(row["type"] for row in selected)),
        "selected_candidate_counts": dict(Counter(len(row["paragraphs"]) for row in selected)),
        "selection_rule": "First min(2000,eligible_count) IDs by ascending SHA256(id), then ID; fixed before any model evaluation",
        "boundaries": "Official dev only; no 2Wiki train data, fitting, checkpoint choice, or hyperparameter tuning. This is a filtered external transfer diagnostic, not official whole-dataset QA evaluation or contamination-free evaluation.",
        "order_policy": "Exactly 2 evidence triples, a unique 3-entity directed path, exactly 2 distinct gold supporting titles, exact string subject-to-unique-context-title matches. Comparison/branching/cyclic/alias-ambiguous/unmatched cases excluded.",
        "independent_audit": {"status": "passed", "eligible_checked": len(eligible), "selected_checked": len(selected), "method": "Separate indegree/outdegree and successor-chain check"},
        "selected_ids_sha256": digest(lock), "examples_sha256": digest(examples_path),
        "metadata_overlap": overlap_report,
        "non_overlap_selection_file": non_overlap_lock.name,
        "non_overlap_selection_ids_sha256": digest(non_overlap_lock),
        "script_sha256": digest(Path(__file__)),
    }
    json_write(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
