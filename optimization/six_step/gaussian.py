"""One-pass paired Gaussian attention experiment. Synthetic examples stay on device."""
import csv
import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
DEVICE = os.environ.get("EXPERIMENT_DEVICE", "cuda")
torch.set_num_threads(2)
torch.set_float32_matmul_precision("highest")
TASK = ["First", "Last", "Parity", "MajProd"]
SUP = ["final", "value", "index"]
OPT = ["sgd", "adam"]
START = ["zero", "small"]
GRID = [2**j for j in range(10, 18)]
BATCH, TEST, VALID, PROBE = 256, 8192, 4096, 32
DEV, CONF = 709101, [709113, 709127, 709139, 709151, 709163]
RATES = {"sgd": [0.1, 0.5], "adam": [0.003, 0.02]}
CELLS = list(itertools.product(range(4), range(3), range(2), range(2)))
THETA = 4.


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def generator(seed, purpose):
    return torch.Generator(device=DEVICE).manual_seed(seed * 100 + purpose)


def frame(seed):
    a = torch.randn(15, 2, device=DEVICE, generator=generator(seed, 1))
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diag(r))


def maj(x, y, z):
    return (x + y + z - x*y*z) / 2


def readouts(v):
    return torch.stack((v[..., 0], v[..., -1], v.prod(-1),
                        maj(*v[..., :3].unbind(-1)) * maj(*v[..., 3:].unbind(-1))), -1)


def predict(U, q, signs, z, soft):
    models, batch = len(U), len(q)
    keys = (z.reshape(-1, 15) @ U).reshape(models, batch, 6, 2, 2)
    state = q[None]
    values, selections = [], []
    for h in range(6):
        a, b = keys[:, :, h].unbind(2)
        score = ((b-a)*state).sum(-1)
        w1 = torch.sigmoid(THETA*score) if soft else (score > 0).to(q.dtype)
        values.append(signs[:, h, 0] + w1*(signs[:, h, 1]-signs[:, h, 0]))
        selections.append(w1)
        state = a + w1[..., None]*(b-a)
    vs, ws = torch.stack(values, -1), torch.stack(selections, -1)
    return readouts(vs), vs, ws


def draw(size, rng, V):
    q = torch.randn(size, 2, generator=rng, device=DEVICE)
    q = torch.nn.functional.normalize(q)
    g = torch.randn(size, 6, 2, generator=rng, device=DEVICE)
    z = torch.randn(size, 6, 2, 15, generator=rng, device=DEVICE)
    s = (g >= 0).float()*2-1
    yy, v, i = predict(V[None], q, s, z, False)
    yy, v, i = yy[0], v[0], i[0]
    flip = (torch.rand(size, generator=rng, device=DEVICE) >= 0.1).float()*2-1
    return q, g, z, s, yy, v[:, 0], i, yy*flip[:, None]


def configs(seed, cells):
    init = torch.randn(15, 2, device=DEVICE, generator=generator(seed, 2))*0.05
    U = torch.stack([init if start else torch.zeros_like(init) for t,s,o,start in cells])
    ids = torch.tensor(cells, device=DEVICE).T
    return U.requires_grad_(True), ids


def model_column(matrix, ids):
    return matrix.gather(-1, ids[:, None, None].expand(-1, matrix.shape[1], 1)).squeeze(-1)


def gradient(U, ids, data):
    q, _, z, signs, _, val, indices, labels = data
    task, sup, _, _ = ids
    pred, v, w = predict(U, q, signs, z, True)
    err = model_column(pred, task) - labels[:, task].T
    loss = err.square()
    loss = loss + (sup == 1)[:, None]*(v[..., 0]-val).square()
    loss = loss + (sup == 2)[:, None]*(2*(w[..., 0]-indices[:, 0])).square()
    return torch.autograd.grad(loss.mean(1).sum()*.5, U)[0]


def info(cell):
    t, s, o, i = cell
    return dict(readout=TASK[t], supervision=SUP[s], optimizer=OPT[o], init=START[i])


def polar(A):
    L, _, R = torch.linalg.svd(A, full_matrices=False)
    return (L @ R) * A.abs().sum((-2, -1)).ne(0)[..., None, None]


@torch.no_grad()
def evaluate(frames, tids, V, test, noisy=False):
    q, _, z, signs, clean, _, gold, labels = test
    scores = []
    for begin in range(0, len(q), 256):
        part = slice(begin, begin+256)
        pred, _, paths = predict(frames, q[part], signs[part], z[part], False)
        error = model_column(pred, tids) != (labels if noisy else clean)[part][:, tids].T
        hit = paths == gold[part]
        scores.append(torch.cat([error.sum(1)[:, None], hit.sum(1),
                                 hit.all(-1).sum(1)[:, None]], 1))
    rates = torch.stack(scores).sum(0).to(torch.float64).cpu().numpy() / len(q)
    native_frame = (frames-V).norm(dim=(-2, -1)).cpu().numpy()/math.sqrt(2)
    space = polar(frames)
    proj = (space@space.transpose(-1,-2)-V@V.T).norm(dim=(-2,-1)).cpu().numpy()/math.sqrt(2)
    identified = frames.abs().sum((-2,-1)).cpu().numpy() > 0
    return [dict(error=x[0], **{f"hop{h+1}": x[h+1] for h in range(6)}, path=x[-1],
                 frame_error=float(native_frame[j]) if identified[j] else None,
                 projector_error=float(proj[j]) if identified[j] else None,
                 identified=int(identified[j])) for j, x in enumerate(rates)]


def r_feature(g, z):
    return ((g[:, 0]-g[:, 1])[:, None] * (z[:, 0]-z[:, 1]))/2


def position_for_last(data):
    b = data[3].mean(-1)
    masses = []
    for h in range(6):
        plus, minus = b.clone(), b.clone()
        plus[:, h], minus[:, h] = 1, -1
        derivative = (readouts(plus)[:, 1] - readouts(minus)[:, 1])/2
        masses.append(float(derivative.square().mean()))
    position = int(np.argmax(masses))
    return position, masses


def moment(data, position):
    q, g, z, signs, _, _, _, labels = data
    b = signs.mean(-1)
    deriv = torch.stack([torch.ones_like(b[:, 0]), torch.zeros_like(b[:, 0]),
                         b[:, 1:].prod(-1), (1-b[:, 1]*b[:, 2])*maj(*b[:, 3:].unbind(-1))/2], 1)
    # Public Gaussian values enter R, and public candidate SIGNS enter derivatives.
    feat = r_feature(g[:, 0], z[:, 0])
    first = torch.einsum("bm,bt,bd->tmd", feat, labels*deriv, q)
    half = len(q)//2
    # First half of each independent batch estimates the subspace. The other half
    # estimates orientation conditional on that subspace at the same checkpoint.
    sub = r_feature(g[:half, position], z[:half, position]).T @ (labels[:half, 1, None]*z[:half, position-1].sum(1))
    orient = (q[half:], z[half:, :position+1], signs[half:, position], labels[half:, 1])
    return first, sub, orient


def terminal_solver(M, blocks):
    _, U = torch.linalg.eigh((M+M.T)/2)
    U = U[:, -2:]
    orient = torch.zeros(2, 2, device=DEVICE)
    for start in range(0, len(blocks), 16):
        q, z, slast, y = [torch.cat(c) for c in zip(*blocks[start:start+16])]
        keys = z @ U
        delta = keys[:, 0, 0]-keys[:, 0, 1]
        values = []
        for initial in (0, 1):
            state = keys[:, 0, initial]
            for h in range(1, z.shape[1]):
                a, b = keys[:, h].unbind(1)
                idx = ((b-a)*state).sum(-1) > 0
                state = torch.where(idx[:, None], b, a)
            values.append(torch.where(idx, slast[:, 1], slast[:, 0]))
        orient += torch.nn.functional.normalize(delta).T @ ((y*(values[0]-values[1]))[:, None]*q)
    return U @ polar(orient)


def probe(seed, V):
    U, ids = configs(seed, CELLS)
    rng = generator(seed, 6)
    samples = torch.stack([gradient(U, ids, draw(BATCH, rng, V)) for _ in range(PROBE)]).double()
    mean = samples.mean(0)
    variance = ((samples-mean).square().sum((-2,-1))).sum(0)/(PROBE-1)
    signal = mean.square().sum((-2,-1)) - variance/PROBE
    ratio = torch.where(variance>0, signal/variance, float("nan")).cpu().numpy()
    return [dict(seed=seed, **info(cell), grads=PROBE, batch=BATCH, mean_norm=mn,
                 noise_rms=math.sqrt(v), unbiased_signal_sq=s,
                 snr_squared=r if math.isfinite(r) else None)
            for cell, mn, v, s, r in zip(CELLS, mean.norm(dim=(-2,-1)).tolist(),
                                          variance.tolist(), signal.tolist(), ratio)]


def trial(seed, chosen=None):
    development = chosen is None
    tic = time.monotonic()
    cells = CELLS*2 if development else CELLS
    rates = [RATES[OPT[cell[2]]][j//len(CELLS)] if development else chosen[j]
             for j, cell in enumerate(cells)]
    U, ids = configs(seed, cells)
    V = frame(seed)
    test = draw(VALID if development else TEST, generator(seed, 4 if development else 5), V)
    rng = generator(seed, 3)
    lr = torch.tensor(rates, device=DEVICE)[:, None, None]
    m, v = torch.zeros_like(U), torch.zeros_like(U)
    adam = ids[2].bool()[:, None, None]
    rows, archive = [], {"V": V.cpu().numpy()}
    F, M, blocks = torch.zeros(4,15,2,device=DEVICE), torch.zeros(15,15,device=DEVICE), []
    for step in range(1, GRID[-1]//BATCH+1):
        data = draw(BATCH, rng, V)
        grad = gradient(U, ids, data)
        with torch.no_grad():
            if not development:
                if step == 1:
                    position, masses = position_for_last(data)
                    assert position > 0
                    write_json(f"position_{seed}.json", dict(index_1based=position+1, masses=masses,
                                                           reused_public_records=BATCH, extra_labels=0))
                f, mm, block = moment(data, position)
                F += f
                M += mm
                blocks.append(block)
            grad /= grad.norm(dim=(-2,-1),keepdim=True).clamp_min(1.)
            m.mul_(.9).add_(grad, alpha=.1)
            v.mul_(.999).addcmul_(grad,grad,value=.001)
            dm = (m/(1-.9**step))/((v/(1-.999**step)).sqrt()+1e-8)
            U -= (lr / math.sqrt(1+(step-1)/128.))*torch.where(adam, dm, grad)
        n = step*BATCH
        if development and n != GRID[-1] or n not in GRID:
            continue
        if development:
            val = evaluate(polar(U.detach()), ids[0], V, test, noisy=True)
            return [dict(**info(cell), rate=rate, noisy_error=x["error"])
                    for cell, rate, x in zip(cells,rates,val)]
        first = polar(F)
        terminal = terminal_solver(M, blocks)
        mats = torch.cat([U.detach(), polar(U.detach()), first, terminal[None]])
        tids = torch.cat([ids[0], ids[0], torch.arange(4,device=DEVICE), ids[0].new_tensor([1])])
        scores = evaluate(mats, tids, V, test)
        archive[f"n{n}"] = mats.cpu().numpy()
        common = dict(seed=seed,n=n)
        for j,(cell,rate) in enumerate(zip(cells, rates)):
            for dep, offset in [("native", 0), ("polar", len(cells))]:
                rows.append(dict(**common,**info(cell),rate=rate,deployment=dep, **scores[j+offset]))
        for t, met in enumerate(scores[2*len(cells):]):
            rows.append(dict(**common, readout=TASK[t if t<4 else 1],supervision="final",
                             optimizer="moment_first" if t<4 else "moment_later",init="none",
                             rate=None,deployment="polar",**met))
    np.savez_compressed(OUT/f"frames_{seed}.npz", **archive)
    write_csv(f"rows_{seed}.csv", rows)
    write_csv(f"gradient_{seed}.csv", probe(seed,V))
    print(f"confirmed {seed} in {time.monotonic()-tic:.1f}s", flush=True)


def main():
    protocol = dict(seed_dev=DEV,seeds_confirm=CONF,grid=GRID,batch=BATCH,test=TEST,
                    validation=VALID,gradient_probe_batches=PROBE,learning_rates=RATES,
                    theta=THETA,device=DEVICE,torch=torch.__version__,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    cells=[info(c) for c in CELLS])
    write_json("protocol.json", protocol)
    print("development", flush=True)
    cache = OUT/"selection.json"
    if cache.exists():
        selected = json.loads(cache.read_text())
    else:
        validation = trial(DEV)
        write_csv("validation.csv", validation)
        selected = []
        for j in range(len(CELLS)):
            choices = [validation[j], validation[j+len(CELLS)]]
            selected.append(min(choices, key=lambda x:x["noisy_error"])["rate"])
        write_json("selection.json", selected)
    print("validation locked",flush=True)
    for seed in CONF:
        if not (OUT/f"gradient_{seed}.csv").exists():
            trial(seed, selected)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
