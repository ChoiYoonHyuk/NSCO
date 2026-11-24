"""Frozen retrieval-trained representation; separate cache, no labels in inputs."""
from pathlib import Path
import hashlib
import json
import os
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".hf_cache")
os.environ["HF_HUB_DISABLE_XET"] = "1"
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download


def main():
    start = time.time()
    out = ROOT / "data/e5"
    out.mkdir(parents=True, exist_ok=True)
    model_id = "intfloat/e5-base-v2"
    meta_file = out / "model_source.json"
    if not meta_file.exists():
        info = json.load(urllib.request.urlopen("https://huggingface.co/api/models/" + model_id))
        meta_file.write_text(json.dumps({"model": model_id, "revision": info["sha"],
                                        "source": "https://huggingface.co/" + model_id}, indent=2))
    source = json.loads(meta_file.read_text())
    path = snapshot_download(model_id, revision=source["revision"],
                             allow_patterns=["*.json", "*.txt", "model.safetensors"],
                             cache_dir=ROOT / ".hf_cache/hub", max_workers=4)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True).to("cuda", dtype=torch.bfloat16).eval()
    torch.set_num_threads(8)
    texts, lookup = [], {}
    def index(text):
        if text not in lookup:
            lookup[text] = len(texts)
            texts.append(text)
        return lookup[text]
    for split in ["train", "dev"]:
        rows = [json.loads(x) for x in (ROOT / f"data/musique_ans_v1.0_{split}.jsonl").open()]
        old = dict(np.load(ROOT / f"data/indices_{split}.npz"))
        q, p = [], []
        for row in rows:
            q.append(index("query: " + row["question"]))
            pars = sorted(row["paragraphs"], key=lambda x: x["idx"])
            p.append([index("passage: " + x["title"] + "\n" + x["paragraph_text"]) for x in pars]
                     + [index("passage: [PAD]")] * (20-len(pars)))
        assert [r["id"] for r in rows] == old["ids"].tolist()
        old["q"], old["p"] = np.array(q), np.array(p)
        np.savez_compressed(out / f"indices_{split}.npz", **old)
    features = np.empty((len(texts), 768), dtype=np.float32)
    truncation = 0
    with torch.inference_mode():
        for i in range(0, len(texts), 96):
            batch = texts[i:i+96]
            raw = tokenizer(batch, truncation=False)
            truncation += sum(len(x) > 512 for x in raw["input_ids"])
            tokens = tokenizer(batch, truncation=True, max_length=512,
                               padding=True, return_tensors="pt").to("cuda")
            hidden = model(**tokens).last_hidden_state.float()
            mean = (hidden * tokens.attention_mask[..., None]).sum(1) / tokens.attention_mask.sum(1)[:, None]
            features[i:i+len(batch)] = F.normalize(mean, dim=-1).cpu().numpy()
            if i % 9600 == 0:
                print(json.dumps({"encoded": i, "total": len(texts), "seconds": time.time()-start}), flush=True)
    np.save(out / "features.npy", features)
    (out / "text_sha256.json").write_text(json.dumps([hashlib.sha256(t.encode()).hexdigest() for t in texts]))
    manifest = {**source, "seconds": time.time()-start, "unique_texts": len(texts),
                "max_length": 512, "truncated_unique_texts": truncation,
                "pooling": "masked mean then L2 normalization; no dataset centering",
                "dtype": "bf16 encoder; float32 pooling and features",
                "fields": ["question", "paragraphs.title", "paragraphs.paragraph_text"],
                "features_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "encoder_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("COMPLETE", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
