"""Depth extension of the six-hop Gaussian small-start Adam experiment."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import gaussian as base

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "plan.json").read_text())
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def parity(frames, query, signs, keys, soft):
    batch, depth = signs.shape[:2]
    projected = (keys.reshape(-1, 15) @ frames).reshape(len(frames), batch, depth, 2, 2)
    state = query[None]
    values, choices = [], []
    for hop in range(depth):
        first, second = projected[:, :, hop].unbind(2)
        delta = second - first
        score = (state * delta).sum(-1)
        weight = torch.sigmoid(base.THETA * score) if soft else (score > 0).to(query.dtype)
        values.append(signs[:, hop, 0] + weight * (signs[:, hop, 1] - signs[:, hop, 0]))
        choices.append(weight)
        state = first + weight[..., None] * delta
    return torch.stack(values, -1).prod(-1), values[0], torch.stack(choices, -1)


def draw(size, depth, rng, teacher):
    q = torch.nn.functional.normalize(torch.randn(size, 2, generator=rng, device=base.DEVICE))
    gauss = torch.randn(size, depth, 2, generator=rng, device=base.DEVICE)
    keys = torch.randn(size, depth, 2, 15, generator=rng, device=base.DEVICE)
    signs = (gauss >= 0).float() * 2 - 1
    hard_y, c1, path = parity(teacher[None], q, signs, keys, False)
    flip = (torch.rand(size, generator=rng, device=base.DEVICE) >= .1).float() * 2 - 1
    return q, signs, keys, hard_y[0], c1[0], path[0], hard_y[0] * flip


def gradient(frames, aux, data):
    q, signs, keys, _, c1, _, y = data
    pred, first, _ = parity(frames, q, signs, keys, True)
    loss = (pred - y).square() + aux[:, None] * (first - c1).square()
    return torch.autograd.grad(.5 * loss.mean(1).sum(), frames)[0]


def validate_against_six_hop():
    seed = 708719
    cells = [(2, 0, 1, 1), (2, 1, 1, 1)]
    frames, ids = base.configs(seed, cells)
    teacher = base.frame(seed)
    reference = base.draw(256, base.generator(seed, 3), teacher)
    current = draw(256, 6, base.generator(seed, 3), teacher)
    for x, y in [(current[0], reference[0]), (current[1], reference[3]),
                 (current[2], reference[2]), (current[3], reference[4][:, 2]),
                 (current[4], reference[5]), (current[5], reference[6]),
                 (current[6], reference[7][:, 2])]:
        assert torch.equal(x, y)
    g0 = base.gradient(frames, ids, reference)
    g1 = gradient(frames, ids[1] == 1, current)
    torch.testing.assert_close(g0, g1, rtol=2e-5, atol=1e-7)
    return (g0-g1).abs().max().item()


@torch.no_grad()
def evaluate(frames, data):
    q, signs, keys, clean, _, path, _ = data
    frames = base.polar(frames)
    error = torch.zeros(len(frames), dtype=torch.int64, device=base.DEVICE)
    agreement = torch.zeros_like(error)
    for begin in range(0, len(q), 1024):
        sl = slice(begin, begin + 1024)
        pred, _, choices = parity(frames, q[sl], signs[sl], keys[sl], False)
        error += (pred != clean[sl]).sum(1)
        agreement += (choices == path[sl]).all(-1).sum(1)
    return error.tolist(), agreement.tolist()


def run(seed, depth):
    configs = [(r, arm) for r in PLAN["rates"] for arm in PLAN["arms"]]
    frames, ids = base.configs(seed, [(2, int(a == "value"), 1, 1) for _, a in configs])
    teacher = base.frame(seed)
    test = draw(PLAN["test_examples"], depth, base.generator(seed, 5), teacher)
    train = base.generator(seed, 3)
    rate = torch.tensor([r for r, _ in configs], device=base.DEVICE)[:, None, None]
    m, v = torch.zeros_like(frames), torch.zeros_like(frames)
    rows = []
    batch = PLAN["batch"]
    for step in range(1, PLAN["max_examples"] // batch + 1):
        grad = gradient(frames, ids[1] == 1, draw(batch, depth, train, teacher))
        with torch.no_grad():
            grad /= grad.norm(dim=(-2,-1), keepdim=True).clamp_min(1.)
            m.mul_(.9).add_(grad, alpha=.1)
            v.mul_(.999).addcmul_(grad, grad, value=.001)
            direction = (m / (1 - .9 ** step)) / ((v / (1 - .999 ** step)).sqrt() + 1e-8)
            frames -= (rate / math.sqrt(1 + (step - 1) / 128.)) * direction
        n = step * batch
        if n in PLAN["grid_examples"]:
            errors, hits = evaluate(frames, test)
            for (r, arm), e, h in zip(configs, errors, hits):
                rows.append(dict(seed=seed, depth=depth, rate=r, arm=arm, n=n,
                                 labels=n * (1 + int(arm == "value")), errors=e,
                                 path_hits=h, test=PLAN["test_examples"]))
    return rows


def main():
    checked = validate_against_six_hop()
    metadata = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                    device=torch.cuda.get_device_name(), reference_gradient_max_abs_diff=checked)
    for filename in ("deep_adam.py", "gaussian.py", "plan.json"):
        metadata[filename] = hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
    (OUT/"provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for seed in PLAN["seeds"]:
        for depth in PLAN["depths"]:
            dest = OUT / f"metrics_{seed}_{depth}.csv"
            if dest.exists():
                continue
            start = time.monotonic()
            rows = run(seed, depth)
            tmp = dest.with_suffix(".tmp")
            with tmp.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            tmp.replace(dest)
            print(f"completed teacher {seed}, depth {depth}, {time.monotonic()-start:.1f}s", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
