"""Prepare the original E5 cache encoder's label metadata from official JSONL."""
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    for split in ("train", "dev"):
        with (DATA / f"musique_ans_v1.0_{split}.jsonl").open() as source:
            rows = [json.loads(line) for line in source]
        labels = [[x["paragraph_support_idx"] for x in r["question_decomposition"]] for r in rows]
        assert all(2 <= len(v) <= 4 and len(set(v)) == len(v) for v in labels)
        counts = [len(r["paragraphs"]) for r in rows]
        assert all(2 <= n <= 20 for n in counts)
        np.savez_compressed(DATA / f"indices_{split}.npz",
                            ids=np.array([r["id"] for r in rows]),
                            labels=np.array([v + [-1] * (4-len(v)) for v in labels]),
                            hops=np.array(list(map(len, labels))), candidate_count=np.array(counts))
    # The upstream encoder reads this revision receipt instead of resolving a mutable head.
    frozen = DATA / "e5"
    frozen.mkdir(exist_ok=True)
    (frozen / "model_source.json").write_text(json.dumps(dict(
        model="intfloat/e5-base-v2", revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        source="https://huggingface.co/intfloat/e5-base-v2"), indent=2))


if __name__ == "__main__":
    main()
