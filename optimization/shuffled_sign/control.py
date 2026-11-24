"""Matched first-sign label assignment at twelve hops with two fixed Adam rates."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import deep_adam as deep
import gaussian as base

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / "control_plan.json").read_text())
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def gradient(frames, targets, use_aux, data):
    q, signs, keys, _, _, _, noisy = data
    parity, chosen, _ = deep.parity(frames, q, signs, keys, True)
    loss = (parity - noisy).square() + use_aux[:, None] * (chosen - targets).square()
    return torch.autograd.grad(.5 * loss.mean(1).sum(), frames)[0]


def run_seed(seed):
    configs = [(rate, arm) for rate in SPEC["rates"] for arm in SPEC["arms"]]
    teacher = base.frame(seed)
    frames, _ = base.configs(seed, [(2, int(arm != "final"), 1, 1) for _, arm in configs])
    assert all(torch.equal(f, frames[0]) for f in frames)
    use_aux = torch.tensor([arm != "final" for _, arm in configs], device=base.DEVICE)
    rates = torch.tensor([rate for rate, _ in configs], device=base.DEVICE)[:, None, None]
    train = base.generator(seed, 3)
    test = deep.draw(SPEC["test"], SPEC["depth"], base.generator(seed, 5), teacher)
    m, v = torch.zeros_like(frames), torch.zeros_like(frames)
    rows = []
    for step in range(1, SPEC["records"] // SPEC["batch"] + 1):
        data = deep.draw(SPEC["batch"], SPEC["depth"], train, teacher)
        true = data[4]
        shifted = true.roll(1)
        if step == 1:
            assert torch.equal(true.sum(), shifted.sum())
        targets = torch.stack([shifted if arm == "shift" else true for _, arm in configs])
        grad = gradient(frames, targets, use_aux, data)
        if step == 1:
            for rate_id in range(len(SPEC["rates"])):
                ix = slice(rate_id * 3, rate_id * 3 + 2)
                reference = deep.gradient(frames[ix], use_aux[ix], data)
                torch.testing.assert_close(grad[ix], reference, atol=1e-7, rtol=2e-5)
        with torch.no_grad():
            grad /= grad.norm(dim=(-2,-1), keepdim=True).clamp_min(1.)
            m.mul_(.9).add_(grad, alpha=.1)
            v.mul_(.999).addcmul_(grad, grad, value=.001)
            direction = (m / (1 - .9 ** step)) / ((v / (1 - .999 ** step)).sqrt() + 1e-8)
            frames -= rates / math.sqrt(1 + (step - 1) / 128.) * direction
        n = step * SPEC["batch"]
        if n in SPEC["checkpoints"]:
            errors, hits = deep.evaluate(frames, test)
            for (rate, arm), e, h in zip(configs, errors, hits):
                rows.append(dict(seed=seed, depth=SPEC["depth"], rate=rate, arm=arm,
                                 examples=n, labels=n*(1 + int(arm != "final")), errors=e,
                                 path_hits=h, test=SPEC["test"]))
    return rows


def main():
    metadata = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                ("control_plan.json","control.py","deep_adam.py","gaussian.py","plan.json")}
    metadata.update(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                    previous_six_hop_regression_max_abs_diff=deep.validate_against_six_hop())
    (OUT/"provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for seed in SPEC["seeds"]:
        filename = OUT / f"teacher_{seed}.csv"
        if filename.exists():
            continue
        start = time.monotonic()
        rows = run_seed(seed)
        with filename.with_suffix(".tmp").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        filename.with_suffix(".tmp").replace(filename)
        print(f"completed {seed} {time.monotonic() - start:.1f}s",flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
