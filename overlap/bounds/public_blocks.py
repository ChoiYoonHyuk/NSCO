"""Numerical evaluation of the public-block construction (Appendix O)."""
import csv
from itertools import product
import math
from pathlib import Path

import numpy as np

OUTPUT = Path("results/overlap_bounds")
DIMENSIONS = (2, 4)
OVERLAPS = (0.25, 0.5, 1.0)
LENGTHS = (2, 3, 4)
SAMPLE_SIZES = (32, 128)
STATES = ("plus_axis", "minus_axis", "perpendicular_axis", "uniform_query")
PUBLIC_INPUTS = 4096
BANK_SEED = 200501
TEST_SEED = 200511


def path(values,private,q,rho):
    """Broadcasted latent coordinates; output is full hard-path parity."""
    z=private.copy();z[...,0]*=math.sqrt(1-rho*rho);z[...,0]+=rho*values
    state=q.copy();out=np.ones(state.shape[:-1],dtype=np.int8)
    for h in range(values.shape[-2]):
        keys=z[...,h,:,:];idx=np.argmax(np.sum(keys*state[...,None,:],axis=-1),axis=-1)
        state=np.take_along_axis(keys,idx[...,None,None],axis=-2)[...,0,:]
        selected=np.take_along_axis(values[...,h,:],idx[...,None],axis=-1)[...,0]
        out*=np.where(selected>=0,1,-1).astype(np.int8)
    return out


def ambient_parity(values,private,query,rho):
    # Ambient coordinates are (public value, independent private record).
    # No production path/gate/statistics function is imported.
    d=private.shape[-1];shape=np.broadcast_shapes(values.shape[:-2],private.shape[:-3],query.shape[:-1])
    frame=np.zeros((d+1,d));frame[0,0]=rho;frame[1,0]=math.sqrt(1-rho*rho)
    frame[np.arange(2,d+1),np.arange(1,d)]=1.
    projector=frame@frame.T
    state=np.broadcast_to(query@frame.T,shape+(d+1,)).copy()
    parity=np.ones(shape,dtype=np.int8)
    for hop in range(values.shape[-2]):
        records=np.empty(shape+(2,d+1))
        records[...,0]=values[...,hop,:]
        records[...,1:]=private[...,hop,:,:]
        scores=np.einsum('...ni,...i->...n',records,state)
        winner=np.argmax(scores,axis=-1)
        selected=np.take_along_axis(records,winner[...,None,None],axis=-2)[...,0,:]
        parity*=np.where(selected[...,0]>=0,1,-1).astype(np.int8)
        state=selected@projector
    return parity


def run_cell(d, rho, length):
    bank = np.random.default_rng(np.random.SeedSequence([BANK_SEED, d, int(rho * 100), length]))
    rng = np.random.default_rng(np.random.SeedSequence([TEST_SEED, d, int(rho * 100), length]))
    largest = max(SAMPLE_SIZES)
    bank_q = bank.standard_normal((largest, d))
    bank_q /= np.linalg.norm(bank_q, axis=1, keepdims=True)
    bank_noise = bank.standard_normal((largest, length, 2, d))
    values = rng.standard_normal((PUBLIC_INPUTS, length, 2))
    private = rng.standard_normal((PUBLIC_INPUTS, length, 2, d))
    second_private = rng.standard_normal((PUBLIC_INPUTS, length, 2, d))
    q = rng.standard_normal((PUBLIC_INPUTS, d))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    second_q = rng.standard_normal((PUBLIC_INPUTS, d))
    second_q /= np.linalg.norm(second_q, axis=1, keepdims=True)
    unit = np.eye(d)
    directions = [np.broadcast_to(u, q.shape) for u in (unit[0], -unit[0], unit[1])] + [q]
    labels = np.stack([path(values, private, u, rho) for u in directions])
    for u, expected in zip(directions, labels):
        np.testing.assert_array_equal(ambient_parity(values, private, u, rho), expected)
    np.testing.assert_array_equal(
        path(values, second_private, second_q, rho), ambient_parity(values, second_private, second_q, rho)
    )

    gates = np.empty((len(SAMPLE_SIZES), PUBLIC_INPUTS))
    checked = 0
    for start in range(0, PUBLIC_INPUTS, 64):
        block = values[start:start + 64, None]
        vv = np.broadcast_to(block, (len(block), largest, length, 2))
        ee = np.broadcast_to(bank_noise, (*vv.shape, d))
        qq = np.broadcast_to(bank_q, (len(block), largest, d))
        plus = path(vv, ee, qq, rho)
        minus = path(-vv, ee, qq, rho)
        np.testing.assert_array_equal(plus, ambient_parity(block, bank_noise[None], bank_q[None], rho))
        np.testing.assert_array_equal(minus, ambient_parity(-block, bank_noise[None], bank_q[None], rho))
        checked += plus.size + minus.size
        for j, n in enumerate(SAMPLE_SIZES):
            hp = plus[:, :n].mean(1)
            hm = minus[:, :n].mean(1)
            gates[j, start:start + len(block)] = (hp + (-1)**length * hm) / 2

    rows = []
    for n, gate in zip(SAMPLE_SIZES, gates):
        signal = labels * gate
        reference = signal[:2].mean(axis=0)
        for name, value in zip(STATES, signal):
            contrast = value - reference
            se = contrast.std(ddof=1) / math.sqrt(PUBLIC_INPUTS)
            z = abs(contrast.mean()) / se if se else (0.0 if contrast.mean() == 0 else math.inf)
            rows.append((d, rho, length, n, name, value.mean(), contrast.mean(), se, z))
    return rows, checked


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    total = 0
    for d, rho, length in product(DIMENSIONS, OVERLAPS, LENGTHS):
        cell, checked = run_cell(d, rho, length)
        rows.extend(cell)
        total += checked
    table = OUTPUT / "public_blocks.csv"
    with table.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("dimension", "overlap", "length", "samples", "state", "signal",
                         "state_difference", "standard_error", "absolute_standard_errors"))
        writer.writerows(rows)
    print(f"{table}: {total:,} independently checked conditional parities; "
          f"maximum absolute state difference in standard errors = {max(row[-1] for row in rows):.6f}")


if __name__ == "__main__":
    main()
