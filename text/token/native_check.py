"""Evaluate the released GPT-2 on the authors' unchanged token task."""
import csv
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "upstream" / "src")]
from models import build_model
from munch import Munch
from tasks import InductionHopsFullSequenceTask

COMMIT = "5735bca9545545774098994e21392b51b285bcc5"
WEIGHTS = "9f69699e7293e691f8780d713327df1e6e243c6d887829f1b2d94abd4f70364b"
SEEDS = (127, 233, 349, 461, 587)


def main():
    assert os.environ.get("PYTHONHASHSEED") == "0", "Upstream samples from Python sets."
    torch.set_num_threads(2)
    folder = HERE / "upstream/models/induction_hops_full_seq/width4-depth6"
    weights = folder / "model_100000.pt"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == WEIGHTS
    conf = yaml.safe_load((folder / "config.yaml").read_text())
    task_args = conf["training"]["task_kwargs"]
    model = build_model(Munch.fromDict(conf["model"]), task_args["seq_len"])
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    model.cuda().eval()
    rows = []
    start = time.perf_counter()
    for seed in SEEDS:
        for hops in range(17):
            task = InductionHopsFullSequenceTask(**task_args, rng=np.random.RandomState(seed * 100 + hops))
            nonnull = task.token_map[task.DOES_NOT_EXIST_CHAR]
            errors = tokens = valid_errors = valid_tokens = 0
            for _ in range(4):
                ids, y = task.get_batch(32, hops=hops)
                with torch.inference_mode():
                    wrong = model(ids.cuda()).argmax(-1).cpu() != y
                valid = y != nonnull
                valid[:, 0] = False
                errors += int(wrong.sum())
                tokens += wrong.numel()
                valid_errors += int(wrong[valid].sum())
                valid_tokens += int(valid.sum())
            rows.append(dict(seed=seed, hops=hops, sequences=128, tokens=tokens, errors=errors,
                             nonnull_char_tokens=valid_tokens, nonnull_char_errors=valid_errors))
        print("native seed", seed, "seconds", round(time.perf_counter() - start, 1), flush=True)
    with (HERE / "native_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
