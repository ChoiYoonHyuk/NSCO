"""Fresh Gaussian core study: public mathematical kernels, no study outcomes."""
from __future__ import annotations
import hashlib
import itertools
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T95_DF5 = 2.570581835636314
READOUTS = ("terminal", "majority_product", "parity")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def generator(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def make_frame(rng, D, d):
    u, r = np.linalg.qr(rng.standard_normal((D - 1, d)))
    return u * np.where(np.diag(r) >= 0, 1., -1.)[None, :]


def draw(rng, n, depth, D, d):
    q = rng.standard_normal((n, d))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    g = rng.standard_normal((n, depth, 2))
    z = rng.standard_normal((n, depth, 2, D - 1))
    return q, g, z


def signs(x):
    return np.where(x >= 0, 1, -1)


def noise(rng, n, contrast):
    return np.where(rng.random(n) < (1 + contrast) / 2, 1, -1)


def hard_path(q, g, z, frame):
    keys = z @ frame
    query, indices, values = q, [], []
    rows = np.arange(len(q))
    for h in range(z.shape[1]):
        idx = np.einsum("bnd,bd->bn", keys[:, h], query).argmax(1)
        indices.append(idx)
        values.append(signs(g[rows, h, idx]))
        query = keys[rows, h, idx]
    return np.stack(indices, 1), np.stack(values, 1)


def soft_path(q, g, z, frame, theta):
    # For arbitrary frame U, keys=zU and the next query=U^T(weighted z).
    # This equals the ambient U U^T recurrence, including its derivatives.
    keys = z @ frame
    query, values = q, []
    for h in range(z.shape[1]):
        scores = theta * np.einsum("bnd,bd->bn", keys[:, h], query)
        weights = np.exp(scores - scores.max(1, keepdims=True))
        weights /= weights.sum(1, keepdims=True)
        values.append((weights * g[:, h]).sum(1))
        query = (weights[:, :, None] * keys[:, h]).sum(1)
    return np.stack(values, 1)


def readout(values, name):
    if name == "terminal":
        return values[..., -1]
    if name == "parity":
        return values.prod(-1)
    assert name == "majority_product" and values.shape[-1] % 3 == 0
    triple = values.reshape(*values.shape[:-1], -1, 3)
    return ((triple.sum(-1) - triple.prod(-1)) / 2).prod(-1)


def derivative_first(g, name, depth):
    b = signs(g[:, :depth]).mean(2)
    if name == "terminal":
        return np.zeros(len(g)) if depth > 1 else np.ones(len(g))
    if name == "parity":
        return b[:, 1:].prod(1)
    assert name == "majority_product" and depth % 3 == 0
    return (1 - b[:, 1] * b[:, 2]) / 2 * (
        readout(b[:, 3:], name) if depth > 3 else np.ones(len(g)))


def kappa_first(name, depth):
    return (0. if depth > 1 else 1.) if name == "terminal" else (
        2. ** (1 - depth) if name == "parity" else (10 / 13) * (13 / 32) ** (depth // 3))


def beta(name, depth):
    return .5 if name == "terminal" else (2. ** -depth if name == "parity" else (13 / 32) ** (depth // 3))


def centered_R(g, z):
    return (g[:, :, None] * (z - z.mean(1, keepdims=True))).sum(1)


def polar(matrix):
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    cutoff = np.finfo(float).eps * max(matrix.shape) * (singular[0] if len(singular) else 0.)
    return left @ right, int((singular > cutoff).sum())


def statistics(values):
    v = np.asarray(values, dtype=float)
    assert len(v) == 6 and np.isfinite(v).all()
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    half = T95_DF5 * sd / math.sqrt(6)
    return {"mean": mean, "sd": sd, "ci95_half_width": half,
            "ci95": [mean - half, mean + half], "seed_values": v.tolist()}


def self_test():
    rng = generator(98765, 1)
    D, d, k, n = 9, 3, 6, 37
    frame = make_frame(rng, D, d)
    q, g, z = draw(rng, n, k, D, d)
    indices, values = hard_path(q, g, z, frame)
    W = np.concatenate([g[..., None], z], -1)
    ambient = np.vstack([np.zeros((1, d)), frame])
    query = q @ ambient.T
    rows = np.arange(n)
    ii, vv = [], []
    for h in range(k):
        ix = (W[:, h] * query[:, None]).sum(-1).argmax(1)
        ii.append(ix); vv.append(signs(W[rows, h, ix, 0]))
        query = W[rows, h, ix] @ ambient @ ambient.T
    assert np.array_equal(indices, np.stack(ii, 1))
    assert np.array_equal(values, np.stack(vv, 1))
    # Non-feasible frames ensure the latent implementation does not rely on U^TU=I.
    f = frame * 1.3
    ambient = np.vstack([np.zeros((1, d)), f])
    query, raw = q @ ambient.T, []
    for h in range(k):
        scores = 3. * (W[:, h] * query[:, None]).sum(-1)
        w = np.exp(scores - scores.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        v = (w[..., None] * W[:, h]).sum(1)
        raw.append(v[:, 0]); query = v @ ambient @ ambient.T
    assert np.allclose(soft_path(q, g, z, f, 3.), np.stack(raw, 1), atol=1e-12)
    bits = np.array(list(itertools.product([-1., 1.], repeat=k)))
    for name in READOUTS:
        target = (bits[:, -1] if name == "terminal" else bits.prod(1) if name == "parity"
                  else np.sign(bits[:, :3].sum(1)) * np.sign(bits[:, 3:].sum(1)))
        assert np.array_equal(readout(bits, name), target)
        assert target.mean() == 0
        coeffs = [np.mean(target * (bits[:, [j for j in range(k) if mask >> j & 1]].prod(1)))
                  for mask in range(1 << k)]
        expected_beta = sum(c*c*2.**(-mask.bit_count()) for mask,c in enumerate(coeffs))
        expected_kappa = sum(c*c*2.**(1-mask.bit_count()) for mask,c in enumerate(coeffs) if mask & 1)
        assert math.isclose(expected_beta, beta(name,k), abs_tol=1e-14)
        assert math.isclose(expected_kappa, kappa_first(name,k), abs_tol=1e-14)
        b = signs(g).mean(2); eps = 1e-6
        plus, minus = b.copy(), b.copy(); plus[:,0] += eps; minus[:,0] -= eps
        assert np.allclose((readout(plus,name)-readout(minus,name))/(2*eps), derivative_first(g,name,k), atol=1e-10)
    rr = centered_R(g[:,0],z[:,0])
    assert np.allclose(rr,(g[:,0,0]-g[:,0,1])[:,None]*(z[:,0,0]-z[:,0,1])/2)
    returned,_ = polar(rng.standard_normal((D-1,d)))
    assert np.allclose(returned.T@returned,np.eye(d),atol=1e-12)
    return {"ambient_latent_hard":True,"ambient_latent_soft_nonfeasible":True,
            "boolean_readouts":True,"fourier_beta_kappa":True,"public_derivatives":True,
            "centered_statistic":True,"polar_constraint":True}
