"""Frozen, all-candidate E5 contexts; gold metadata never enters encoder inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(dataset, split):
    assert dataset == "2wiki" and split == "dev"
    source = ROOT / "data/wiki_transfer/examples.jsonl"
    full = [json.loads(x) for x in source.open()]
    rows = [{k:r[k] for k in ["id","question","candidates"]} for r in full]
    texts, lookup = [], {}
    def index(text):
        if text not in lookup:
            lookup[text]=len(texts)
            texts.append(text)
        return lookup[text]
    width=max(len(r["candidates"]) for r in rows)
    assert len(rows)==1563 and width==10
    idx={"ids":np.asarray([r["id"] for r in full]),
         "q":np.asarray([index("query: "+r["question"]) for r in rows]),
         "p":np.asarray([[index("passage: "+c["title"]+"\n"+c["text"]) for c in r["candidates"]]+[index("passage: [PAD]")]*(width-len(r["candidates"])) for r in rows]),
         "labels":np.asarray([r["gold_support_indices"] for r in full]),
         "hops":np.full(len(rows),2),
         "candidate_count":np.asarray([len(r["candidates"]) for r in rows]),
         "types":np.asarray([r["type"] for r in full])}
    assert idx["ids"].tolist()==json.loads((ROOT/"data/wiki_transfer/selection_ids.json").read_text())
    out=ROOT/"data/transfer_contexts/2wiki"
    return source, rows, idx, out, out/"features.npy", texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["2wiki"], required=True)
    ap.add_argument("--split", choices=["dev"], default="dev")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    source, rows, idx, out, feature_path, base_texts = prepare(args.dataset, args.split)
    out.mkdir(parents=True, exist_ok=True)
    lock = {"script_sha256": sha(__file__), "source_sha256": sha(source), "revision": REVISION,
            "dataset": args.dataset, "split": args.split, "batch_size": args.batch_size,
            "context_template": "query: {question}\\n{candidate.title}\\n{candidate.text}",
            "max_length": 512, "preparation_manifest_sha256":sha(ROOT/"data/wiki_transfer/manifest.json"), "selection_ids_sha256":sha(ROOT/"data/wiki_transfer/selection_ids.json")}
    lock_path = out / f"encoding_lock_{args.split}.json"
    if lock_path.exists():
        assert json.loads(lock_path.read_text()) == lock, "Context execution lock changed"
    else:
        lock_path.write_text(json.dumps(lock, indent=2))
    manifest_path = out / f"encoder_manifest_{args.split}.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())["lock"] == lock
        print("ALREADY COMPLETE", manifest_path)
        return
    os.environ["HF_HOME"] = str(ROOT / ".hf_cache")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from huggingface_hub import snapshot_download
    torch.set_num_threads(6)
    path = snapshot_download("intfloat/e5-base-v2", revision=REVISION,
                             cache_dir=ROOT / ".hf_cache/hub", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True).to("cuda", dtype=torch.bfloat16).eval()
    start = time.time()
    def encode(texts, label):
        features = np.empty((len(texts), 768), dtype=np.float32)
        truncated = 0
        with torch.inference_mode():
            for begin in range(0, len(texts), args.batch_size):
                batch = texts[begin:begin + args.batch_size]
                raw = tokenizer(batch, truncation=False)
                truncated += sum(len(ids) > 512 for ids in raw["input_ids"])
                tokens = tokenizer(batch, truncation=True, max_length=512,
                                   padding=True, return_tensors="pt").to("cuda")
                hidden = model(**tokens).last_hidden_state.float()
                mean = (hidden * tokens.attention_mask[..., None]).sum(1) / tokens.attention_mask.sum(1)[:, None]
                features[begin:begin+len(batch)] = F.normalize(mean, dim=-1).cpu().numpy()
                if begin % (args.batch_size * 64) == 0:
                    print(json.dumps({"stage": label, "encoded": begin, "total": len(texts),
                                      "seconds": time.time()-start}), flush=True)
        return features, truncated
    base_truncated = None
    if base_texts is not None:
        features, base_truncated = encode(base_texts, "base")
        np.save(feature_path, features)
        del features
    n = len(rows)
    width = idx["p"].shape[1]
    coords, texts = [], []
    text_hash = hashlib.sha256()
    for i, row in enumerate(rows):
        assert set(row) == {"id", "question", "candidates"}
        for j, c in enumerate(row["candidates"]):
            assert set(c) == {"title", "text"}
            text = "query: " + row["question"] + "\n" + c["title"] + "\n" + c["text"]
            texts.append(text)
            coords.append((i, j))
            text_hash.update(text.encode())
            text_hash.update(b"\0")
    encoded, truncated = encode(texts, "all_candidate_contexts")
    context_path = out / f"context_{args.split}.npy"
    context = np.lib.format.open_memmap(context_path, mode="w+", dtype=np.float32, shape=(n, width, 768))
    context[:] = 0
    ii, jj = np.asarray(coords).T
    context[ii, jj] = encoded
    context.flush()
    del context, encoded
    np.savez_compressed(out / f"indices_{args.split}.npz", **idx)
    manifest = {"status": "complete", "lock": lock, "n": n, "width": width,
                "contexts": len(texts), "context_text_sequence_sha256": text_hash.hexdigest(),
                "context_file_sha256": sha(context_path), "base_feature_file_sha256": sha(feature_path),
                "indices_sha256": sha(out / f"indices_{args.split}.npz"),
                "context_truncated": truncated, "base_truncated": base_truncated,
                "input_fields": ["question", "candidate.title", "candidate.text"],
                "gold_prefix_used": False, "pooling": "bf16 encoder; masked float32 mean; L2 normalization",
                "seconds": time.time()-start}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("COMPLETE", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
