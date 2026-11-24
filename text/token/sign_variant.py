"""Online learning on an independent-sign variant of Sanford et al.'s hop_3."""
import csv
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from native_check import HERE, SEEDS, InductionHopsFullSequenceTask

READOUTS = ("first", "last", "parity", "majority")
AUX = ("final", "value", "index")
ARMS = [(g, a) for g in READOUTS for a in AUX]
BUDGETS = (0, 128, 512, 2048, 8192)
LENGTH = 98


def paths_for(tokens):
    """One pointer jumps to the successor of the last previous equal token."""
    n, length = tokens.shape
    previous = np.full((n, 4), -1, dtype=int)
    pointers = np.empty_like(tokens)
    rows = np.arange(n)
    for t in range(length):
        last = previous[rows, tokens[:, t]]
        pointers[:, t] = np.where(last < 0, -1, last + 1)
        previous[rows, tokens[:, t]] = t
    path = [pointers]
    for _ in range(2):
        p = path[-1]
        path.append(np.where(p < 0, -1, pointers[rows[:, None], p.clip(min=0)]))
    return path


def records(rng, n, native_tokens=None):
    # A uniform first token and independent uniform offsets 1..3 produce
    # the same no-repeat sequence law as the official generator.
    if native_tokens is None:
        changes = rng.randint(1, 4, size=(n, LENGTH))
        changes[:, 0] = rng.randint(4, size=n)
        tokens = np.cumsum(changes, axis=1) % 4
    else:
        tokens = native_tokens
    path = paths_for(tokens)
    valid = path[-1] >= 0
    assert valid.any(axis=1).all()
    query = np.argmax(np.where(valid, rng.rand(n, LENGTH), -1), axis=1)
    rows = np.arange(n)
    truth = np.stack([level[rows, query] for level in path], axis=1)
    assert (truth[:, 0] < query).all() and (np.diff(truth, axis=1) < 0).all()
    signs = rng.randint(2, size=(n, LENGTH)) * 2 - 1
    return [torch.as_tensor(x, device="cuda") for x in (tokens, query, truth, signs)]


def eval_records(seed):
    # The first 128 benchmark sequences match those evaluated for hop 3
    # by native_check.py. Check every pointer against official metadata.
    task = InductionHopsFullSequenceTask(100, 4, 0, 16, np.random.RandomState(seed * 100 + 3))
    chunks = []
    for _ in range(16):
        ids, _, _, _, _, metadata = task.get_batch(32, metadata=True, hops=3)
        tokens = ids[:, 1:].numpy() - task.induction_tokens
        p = paths_for(tokens)
        for j in range(3):
            official = np.asarray([m[j+1][:LENGTH] for m in metadata])
            np.testing.assert_array_equal(p[j], official)
        chunks.append(tokens)
    return records(np.random.RandomState(seed + 200000), 512, np.concatenate(chunks))


def signs_to_labels(s):
    return torch.stack((s[:, 0], s[:, -1], s.prod(-1), s.sum(-1).sign()), dim=1)


class Router(torch.nn.Module):
    """Independent four-by-four match tables and scalar recency per arm."""
    def __init__(self, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        base = .02 * torch.randn(1, 4, 4, generator=gen)
        self.match = torch.nn.Parameter(base.repeat(len(ARMS), 1, 1))
        self.decay = torch.nn.Parameter(torch.full((len(ARMS),), 0.0))
        pos = torch.arange(LENGTH)
        lag = pos[:, None] - pos[None, :]
        self.register_buffer("dist", torch.log1p(lag.clamp_min(0)).float())
        mask = (lag <= 0) | (pos[None, :] == 0)
        # Positions zero and one have no valid earlier successor. Their
        # fallback to zero keeps soft recurrences finite off the true path.
        mask[:2, 0] = False
        self.register_buffer("mask", mask)
        self.register_buffer("arm", torch.arange(len(ARMS))[:, None, None, None])
        self.register_buffer("g", torch.tensor([READOUTS.index(g) for g, _ in ARMS]))

    def forward(self, data, hard=False):
        ids, q, truth, signs = data
        pred_keys = torch.cat((ids[:, :1], ids[:, :-1]), dim=1)
        scores = self.match[self.arm, ids[None, :, :, None], pred_keys[None, :, None, :]]
        scores = scores - F.softplus(self.decay)[:, None, None, None] * self.dist
        scores.masked_fill_(self.mask, -1e4)
        a = torch.arange(len(ARMS), device=ids.device)[:, None]
        b = torch.arange(len(q), device=ids.device)[None, :]
        gold = signs_to_labels(signs.gather(1, truth)).T[self.g]
        if hard:
            pointers = scores.argmax(-1)
            t = q.expand(len(ARMS), -1)
            path = []
            for _ in range(3):
                t = pointers[a, b, t]
                path.append(t)
            path = torch.stack(path, dim=-1)
            selected = signs[b[:, :, None], path]
            guesses = torch.stack((selected[:, :, 0], selected[:, :, -1],
                                   selected.prod(-1), selected.sum(-1).sign()), dim=0)
            correct = path == truth
            return torch.stack(((guesses[self.g, a[:, 0]] != gold).float().mean(1),
                                correct[:, :, 0].float().mean(1), correct.all(-1).float().mean(1)), dim=1)
        prob = scores.softmax(-1)
        val = signs.float()[None, :, :, None]
        c1 = prob @ val
        c2 = prob @ c1
        c3 = prob @ c2
        product = prob @ (val * (prob @ (val * c1)))
        mean = (c1 + c2 + c3 - product) * .5
        arms = torch.stack([v.squeeze(-1)[a, b, q] for v in (c1, c3, product, mean)])
        pred = arms[self.g, a[:, 0]]
        def bitloss(m, y):
            return -((1 + m * y).mul(.5).clamp_min(1e-7).log()).mean(1)
        total = bitloss(pred, gold)
        y1 = signs[b, truth[:, 0]]
        value_loss = bitloss(arms[0], y1)
        index_loss = -prob[a, b, q, truth[:, 0]].clamp_min(1e-8).log().mean(1)
        # Average duplicate losses to keep First+value exactly paired with First.
        return torch.stack([total[i] if aux == "final" else
                            .5 * (total[i] + (value_loss if aux == "value" else index_loss)[i])
                            for i, (_, aux) in enumerate(ARMS)]).sum()


def main():
    torch.set_num_threads(2)
    rows = []
    for seed in SEEDS:
        start = time.perf_counter()
        test = eval_records(seed)
        rng = np.random.RandomState(500000 + seed)
        model = Router(seed).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.035)
        for seen in range(0, BUDGETS[-1] + 1, 32):
            if seen in BUDGETS:
                with torch.inference_mode():
                    metrics = torch.stack([model([v[j:j+32] for v in test], hard=True)
                                           for j in range(0, 512, 32)]).mean(0).cpu().tolist()
                for (g, aux), (error, hop, path) in zip(ARMS, metrics):
                    rows.append(dict(seed=seed, readout=g, supervision=aux, examples=seen,
                                     test_examples=512, sign_error=error, first_accuracy=hop,
                                     path_accuracy=path, seconds_since_seed_start=time.perf_counter()-start))
            if seen < BUDGETS[-1]:
                optimizer.zero_grad()
                model(records(rng, 32)).backward()
                optimizer.step()
        print("sign seed", seed, "seconds", round(time.perf_counter()-start, 1), flush=True)
        with (HERE / "sign_results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
