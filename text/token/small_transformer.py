"""Independent randomly initialized causal Transformers for the four sign tasks."""
import csv
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sign_variant import BUDGETS, HERE, LENGTH, READOUTS, SEEDS, eval_records, records, signs_to_labels


class CausalBlock(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.attn_norm = nn.LayerNorm(width)
        self.ff_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(b, n, 3, 4, 8).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(attn.transpose(1, 2).reshape_as(x))
        return x + self.ff(self.ff_norm(x))


class SmallTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(4, 32)
        self.sign = nn.Embedding(2, 32)
        self.position = nn.Parameter(.02 * torch.randn(LENGTH, 32))
        self.blocks = nn.Sequential(*[CausalBlock() for _ in range(3)])
        self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 1))

    def forward(self, data):
        tokens, queries, _, signs = data
        x = self.token(tokens) + self.sign((signs > 0).long()) + self.position
        h = self.blocks(x)[torch.arange(len(queries), device=x.device), queries]
        return self.head(h).squeeze(-1)


def labels(data):
    return signs_to_labels(data[-1].gather(1, data[2])) > 0


def main():
    torch.set_num_threads(2)
    rows = []
    for seed in SEEDS:
        test = eval_records(seed)
        for gi, g in enumerate(READOUTS):
            start = time.perf_counter()
            torch.manual_seed(seed)
            rng = np.random.RandomState(500000 + seed)
            model = SmallTransformer().cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
            for seen in range(0, BUDGETS[-1]+1, 32):
                if seen in BUDGETS:
                    with torch.inference_mode():
                        pred = torch.cat([model([v[j:j+32] for v in test])
                                          for j in range(0, len(test[0]), 32)])
                        err = (pred.ge(0) != labels(test)[:, gi]).float().mean().item()
                    rows.append(dict(seed=seed, readout=g, examples=seen, test_examples=len(pred),
                                     sign_error=err, parameters=sum(p.numel() for p in model.parameters()),
                                     seconds_since_start=time.perf_counter()-start))
                if seen < BUDGETS[-1]:
                    data = records(rng, 32)
                    optimizer.zero_grad()
                    loss = F.binary_cross_entropy_with_logits(model(data), labels(data)[:, gi].float())
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
            print(seed, g, round(err, 4), flush=True)
            with (HERE / "transformer_results.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
