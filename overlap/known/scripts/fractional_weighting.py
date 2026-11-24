"""Fixed final-label certificate study; all data come from ambient Gaussian paths."""
from pathlib import Path
import argparse
import itertools
import json
import math
import time
import numpy as np
from first_sign import target, query, hard, quadrature, invert, gen, sha, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/fractional_weighting'


def value_readout(v, kind):
    if kind == 'parity':
        return np.prod(v, axis=-1)
    t = v.reshape(*v.shape[:-1], -1, 3)
    return np.prod((t.sum(-1) - t.prod(-1)) / 2, axis=-1)


def derivative(b, kind):
    if kind == 'parity':
        return b[:, 1:].prod(1)
    return (1 - b[:, 1] * b[:, 2]) / 2 * value_readout(b[:, 3:], kind)


def masses(kind, k):
    if kind == 'parity':
        return 2.**(1-k), 2.**(1-k)
    return .125*(5/16)**(k//3-1), (10/13)*(13/32)**(k//3)


def self_test():
    rng = gen(31999, 1)
    # Exhaust every ternary derivative configuration, up to nine positions.
    checks = 0
    for k in (3, 6, 9):
        b = np.array(list(itertools.product([-1., 0., 1.], repeat=k-1)))
        prob = np.prod(np.where(b == 0, .5, .25), axis=1)
        bb = np.column_stack([np.zeros(len(b)), b])
        for kind in ('parity', 'majority_product'):
            d = derivative(bb, kind)
            p, kap = masses(kind, k)
            assert abs(prob @ (np.abs(d) == 1) - p) < 1e-12
            assert abs(prob @ d**2 - kap) < 1e-12
            # Certificate is tested against every compatible sign completion.
            for row, sign in zip(b[np.abs(d) == 1], d[np.abs(d) == 1]):
                missing = np.flatnonzero(row == 0)
                for bits in itertools.product([-1., 1.], repeat=len(missing)):
                    c = row.copy(); c[missing] = bits
                    for first in (-1., 1.):
                        assert value_readout(np.r_[first, c], kind) == sign*first
                        checks += 1
    # Latent and independent ambient recurrence agree for nonzero/full overlap.
    Q = query(rng, 91, 4); W = rng.normal(size=(91, 9, 2, 16))
    B, rr = np.linalg.qr(rng.normal(size=(15, 4)))
    B *= np.where(np.diag(rr) >= 0, 1., -1.)
    rows = np.arange(len(Q))
    for rho in (0., .5, 1.):
        P = target(B, rho); ii, cc = hard(Q, W, P)
        state = Q @ P.T
        for h in range(9):
            idx = (W[:, h] * state[:, None, :]).sum(-1).argmax(1)
            assert np.array_equal(idx, ii[:, h])
            state = W[rows, h, idx] @ (P @ P.T)
        b = np.where(W[..., 0] >= 0, 1., -1.).mean(2)
        for k in (3, 6, 9):
            for kind in ('parity', 'majority_product'):
                dd = derivative(b[:, :k], kind); accepted = abs(dd) == 1
                assert np.array_equal((value_readout(cc[:, :k], kind)*dd)[accepted], cc[accepted, 0])
    return {'status': 'passed', 'compatible_certificate_completions': checks,
            'ambient_recurrence_examples': 273, 'exact_derivative_mass_enumeration': True}


def run():
    cfg = json.loads((ROOT/'fractional_weighting_protocol.json').read_text())
    sources = ['fractional_weighting_protocol.json', 'scripts/fractional_weighting.py', 'scripts/first_sign.py']
    lock = {p: sha(ROOT/p) for p in sources}
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'execution_lock.json').exists(), 'Study already started; preserve evidence.'
    write(OUT/'execution_lock.json', lock)
    write(OUT/'self_test.json', self_test())
    D, d = cfg['D'], cfg['d']; x, w = quadrature(d)
    conditions = [(kind, k) for kind in cfg['readouts'] for k in cfg['depths']]
    nr, nc = len(cfg['overlaps']), len(conditions)
    records = []; started = time.monotonic()
    for seed in cfg['seeds']:
        fr = gen(seed, cfg['streams']['frame']); tr = gen(seed, cfg['streams']['training'])
        te = gen(seed, cfg['streams']['test'])
        B, rr = np.linalg.qr(fr.normal(size=(D-1, d))); B *= np.where(np.diag(rr)>=0, 1., -1.)
        targets = np.stack([target(B, rho) for rho in cfg['overlaps']])
        qt = query(te, cfg['test_count'], d); Wt = te.normal(size=(len(qt), 9, 2, D))
        truths = [hard(qt, Wt, P) for P in targets]
        np.savez_compressed(OUT/f'seed_{seed}_test.npz', q=qt, W=Wt, frames=targets,
                            indices=[z[0] for z in truths], values=[z[1] for z in truths])
        sums = np.zeros((2, nr, nc, D, d)); vecs = np.zeros((2, nr, nc, d))
        counts = np.zeros(nc, dtype=np.int64); total = 0; arrays = {}; states = []
        identity_checks = 0
        for n in cfg['sample_counts']:
            while total < n:
                states.append({'before': total, 'state': tr.bit_generator.state})
                batch = min(cfg['batch_size'], n-total)
                q = query(tr, batch, d); W = tr.normal(size=(batch, 9, 2, D))
                noise = np.where(tr.random(batch)<(1+cfg['lambda'])/2, 1., -1.)
                b = np.where(W[..., 0]>=0, 1., -1.).mean(2)
                ds = np.stack([derivative(b[:, :k], kind) for kind, k in conditions], 1)
                accepted = abs(ds) == 1
                counts += accepted.sum(0)
                delta = (W[:, 0, 0] - W[:, 0, 1])/math.sqrt(2)
                R = delta[:, 0, None]*delta; R[:, 0] = 0
                if total == 0:
                    np.savez_compressed(OUT/f'seed_{seed}_fixture.npz', q=q[:64], W=W[:64], noise=noise[:64])
                for ri, P in enumerate(targets):
                    _, c = hard(q, W, P)
                    ys = np.stack([noise*value_readout(c[:, :k], kind) for kind, k in conditions], 1)
                    weighted = ys*ds
                    correct = noise*c[:, 0]
                    assert np.array_equal(weighted[accepted], np.broadcast_to(correct[:,None], weighted.shape)[accepted])
                    identity_checks += int(accepted.sum())
                    gates = np.stack([weighted*accepted, weighted])
                    vecs[:, ri] += np.einsum('mbc,bd->mcd', gates, q, optimize=True)
                    sums[:, ri] += np.einsum('mbc,bD,bd->mcDd', gates, R, q, optimize=True)
                total += batch
            arrays[f'n{n}_sums'] = sums.copy(); arrays[f'n{n}_vecs'] = vecs.copy()
            arrays[f'n{n}_counts'] = counts.copy()
            for ri, (rho, P) in enumerate(zip(cfg['overlaps'], targets)):
                for ci, (kind, k) in enumerate(conditions):
                    p, kap = masses(kind, k)
                    for mi, method in enumerate(('certificate', 'fractional_control')):
                        denominator = (counts[ci] if mi == 0 else n*kap)*cfg['lambda']
                        zero_acceptance = bool(denominator == 0)
                        if zero_acceptance:
                            frame = np.eye(D, d); r = frame[0].copy()
                        else:
                            frame, r, _, _ = invert(vecs[mi, ri, ci]/denominator, sums[mi, ri, ci]/denominator, x, w)
                        idx, values = hard(qt, Wt[:, :k], frame)
                        prediction = value_readout(values, kind).astype(np.int8)
                        label = value_readout(truths[ri][1][:, :k], kind).astype(np.int8)
                        key = f'n{n}_rho{ri}_cond{ci}_{method}'
                        arrays[key+'_frame'] = frame; arrays[key+'_prediction'] = prediction
                        arrays[key+'_indices'] = idx.astype(np.int8)
                        records.append({'seed': seed, 'n': n, 'rho': rho, 'readout': kind, 'k': k,
                            'method': method, 'certificate_count': int(counts[ci]), 'p_certificate': p,
                            'kappa': kap, 'zero_acceptance': zero_acceptance, 'frame_error': float(np.linalg.norm(frame-P)/math.sqrt(d)),
                            'hard_error': float(np.mean(prediction != label)), 'estimated_rho': float(np.linalg.norm(r))})
        states.append({'before': total, 'state': tr.bit_generator.state})
        np.savez_compressed(OUT/f'seed_{seed}_estimates.npz', **arrays)
        write(OUT/f'seed_{seed}_rng.json', states)
        write(OUT/f'seed_{seed}.json', [r for r in records if r['seed']==seed])
        write(OUT/f'seed_{seed}_checks.json', {'accepted_label_identities_checked': identity_checks})
        print(json.dumps({'seed_complete': seed, 'seconds': time.monotonic()-started}), flush=True)
    assert all(sha(ROOT/p)==h for p,h in lock.items())
    write(OUT/'raw_results.json', records)
    outputs = {p.relative_to(OUT).as_posix(): sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix != '.log'}
    write(OUT/'run_manifest.json', {'status':'complete', 'records':len(records),
        'seconds':time.monotonic()-started, 'numpy':np.__version__, 'python':__import__('sys').version,
        'device':'CPU', 'output_sha256':outputs, 'selection_performed':False})


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--run',action='store_true')
    args=ap.parse_args()
    if args.self_test: print(json.dumps(self_test()))
    elif args.run: run()
    else: ap.error('choose --self-test or --run')
