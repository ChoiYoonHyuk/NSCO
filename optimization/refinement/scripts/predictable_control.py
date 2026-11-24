"""Predictable control variates for averaged SGD."""
import numpy as np
import sample_split as base


def predictable_polar(states):
    result = np.zeros_like(states)
    active = np.linalg.norm(states, axis=(-2, -1)) > 0
    if np.any(active):
        left, _, right = np.linalg.svd(states[active], full_matrices=False)
        result[active] = left @ right
    return result, active


def condition_mask(states, threshold):
    singular=np.linalg.svd(states,compute_uv=False)
    ratio=np.divide(singular[...,-1],singular[...,0],out=np.zeros_like(singular[...,0]),where=singular[...,0]>0)
    return ratio>threshold,ratio


def controlled_sums(q, W, Y, names, Q, active, mu, lam, P=None):
    b = base.signs(W[..., 0]).mean(2)
    e = (base.signs(W[:, 0, 0, 0]) - base.signs(W[:, 0, 1, 0])) / 2
    delta = W[:, 0, 0, 1:] - W[:, 0, 1, 1:]
    derivative = np.stack([base.derivative(b, name) for name in names])
    a = Y * derivative * e
    control = base.signs(np.einsum('bi,rjid,bd->rjb', delta, Q, q))
    bb = control * derivative[None]**2 * e[None, None]**2 * active[..., None]
    A = np.einsum('jb,bi,bd->jid', a, delta, q)
    B = np.einsum('rjb,bi,bd->rjid', bb, delta, q)
    Z = A[None] - lam * B + len(q) * lam * mu[None, :, None, None] * Q
    diagnostics = None
    if P is not None:
        norm = (delta**2).sum(1)
        center = lam * mu[None, :, None, None] * (Q - P[None, None])
        c = a[None] - lam * bb
        cc = A[None] - lam * B
        cvsq = (c*c*norm).sum(-1) + 2*(cc*center).sum((-2, -1)) + len(q)*(center*center).sum((-2, -1))
        plaincenter = -lam * mu[:, None, None] * P[None]
        plainsq = (a*a*norm).sum(-1) + 2*(A*plaincenter).sum((-2, -1)) + len(q)*(plaincenter*plaincenter).sum((-2, -1))
        diagnostics = (plainsq, cvsq)
    return A, B, Z, diagnostics
