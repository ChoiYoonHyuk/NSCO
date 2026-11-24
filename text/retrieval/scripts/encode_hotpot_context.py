"""Pad variable Hotpot candidate counts; reuse the frozen context encoder."""
from pathlib import Path
import json
import numpy as np
import encode_context as helper

ROOT = Path(__file__).resolve().parents[1]


def prepare(dataset, split):
    assert dataset == "hotpot" and split == "dev"
    source = ROOT / "data/hotpot/examples.jsonl"
    full = [json.loads(x) for x in source.open()]
    rows = [{k:r[k] for k in ["id","question","candidates"]} for r in full]
    texts, lookup = [], {}
    def index(text):
        if text not in lookup:
            lookup[text] = len(texts)
            texts.append(text)
        return lookup[text]
    width = max(len(r["candidates"]) for r in rows)
    assert len(rows) == 7405 and width == 10
    idx = {"ids":np.asarray([r["id"] for r in full]),
           "q":np.asarray([index("query: "+r["question"]) for r in rows]),
           "p":np.asarray([[index("passage: "+c["title"]+"\n"+c["text"]) for c in r["candidates"]]
                            +[index("passage: [PAD]")]*(width-len(r["candidates"])) for r in rows]),
           "labels":np.asarray([r["gold_support_indices"] for r in full]),
           "hops":np.full(len(rows),2),
           "candidate_count":np.asarray([len(r["candidates"]) for r in rows]),
           "types":np.asarray([r["type"] for r in full])}
    out = ROOT / "data/contexts/hotpot"
    return source, rows, idx, out, out/"features.npy", texts


def main():
    out = ROOT / "data/contexts/hotpot"
    out.mkdir(parents=True,exist_ok=True)
    lock = {"adapter_script_sha256":helper.sha(__file__),
            "encoder_script_sha256":helper.sha(ROOT/"scripts/encode_context.py"),
            "preparation_manifest_sha256":helper.sha(ROOT/"data/hotpot/manifest.json"),
            "example_sha256":helper.sha(ROOT/"data/hotpot/examples.jsonl"),
            "fields":["question","candidate.title","candidate.text"],
            "variable_candidates_padded_to":10}
    path=out/"adapter_lock.json"
    if path.exists():
        assert json.loads(path.read_text()) == lock
    else:
        path.write_text(json.dumps(lock,indent=2))
    helper.prepare=prepare
    helper.main()
    manifest_path=out/"encoder_manifest_dev.json"
    manifest=json.loads(manifest_path.read_text())
    manifest["preparation_adapter_lock"]=lock
    manifest["gold_order_is_unordered"]=True
    manifest_path.write_text(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
