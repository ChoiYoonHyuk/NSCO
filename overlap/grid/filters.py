"""Gaussian sign and magnitude filters and their finite-support identities."""
from pathlib import Path
from fractions import Fraction as Q
from itertools import product
from functools import lru_cache
from decimal import Decimal, localcontext
import hashlib
import json
import math
import time
import numpy as np

erf = np.vectorize(math.erf, otypes=[float])

def mass(n, eta, gap, complete=True, magnitude=0.):
    values = [1.]
    for length in range(1, n + 1):
        value = .5 * values[-1]
        for j in range(min(gap, length - 2) + 1):
            value += 2.**(j + int(complete)) * eta[j]**2 * values[length - j - 2]
        if length >= 2:
            value += magnitude**2 * values[length-2]
        values.append(value)
    return values[-1]


def gate(b, eta, gap, complete=True, psi=None, magnitude=0.):
    count, n = b.shape
    f = [None] * (n + 1)
    f[n] = np.ones(count)
    for start in range(n - 1, -1, -1):
        value = b[:, start] * f[start + 1]
        middle = np.ones(count)
        for j in range(min(gap, n - start - 2) + 1):
            right = start + j + 1
            endpoint = 2 * (1 - b[:, right]**2) if complete else 1.
            value = value + 2.**j * eta[j] * middle * endpoint * f[right + 1]
            middle *= b[:, right]
        if psi is not None and start + 1 < n:
            value += magnitude * psi[:, start+1] * f[start+2]
        f[start] = value
    return f[0]


def exact_checks(maximum):
    eta = [Q(1, 8 * 4**j) for j in range(maximum)]
    rows = []
    for gap in (0, 1, maximum):
        @lru_cache(None)
        def polynomial(n):
            if n == 0:
                return {(): Q(1)}
            result = {('B',) + word: coef for word, coef in polynomial(n - 1).items()}
            for j in range(min(gap, n - 2) + 1):
                for word, coef in polynomial(n - j - 2).items():
                    for right, factor in [('K', Q(2**j) * eta[j]), ('D', -Q(2**(j+1)) * eta[j])]:
                        key = ('K',) + ('B',) * j + (right,) + word
                        assert key not in result
                        result[key] = factor * coef
            return result
        recurrence = [Q(1)]
        for n in range(1, maximum + 1):
            recurrence.append(recurrence[-1] / 2 + sum(
                (Q(2**(j+1)) * eta[j]**2 * recurrence[n-j-2]
                 for j in range(min(gap, n - 2) + 1)), Q(0)))
            poly = polynomial(n)
            norm = Q(0)
            for word, coefficient in poly.items():
                nonb = [i for i, letter in enumerate(word) if letter != 'B']
                assert len(nonb) % 2 == 0
                moment = Q(1)
                outside = word.count('B')
                for left, right in zip(nonb[::2], nonb[1::2]):
                    assert word[left] == 'K'
                    j = right - left - 1
                    outside -= j
                    moment *= eta[j] * (Q(-1, 2) if word[right] == 'D' else 1)
                moment *= Q(1, 2)**outside
                squared_norm = Q(1, 2)**word.count('B') * Q(1, 4)**word.count('D')
                assert coefficient == moment / squared_norm
                norm += coefficient**2 * squared_norm
            assert norm == recurrence[n]
            if gap == maximum:
                for word in product(('K', 'B', 'D'), repeat=n):
                    nonb = [i for i, letter in enumerate(word) if letter != 'B']
                    nonzero = len(nonb) % 2 == 0 and all(word[i] == 'K' for i in nonb[::2])
                    assert (word in poly) == nonzero
            rows.append({'n': n, 'gap': gap, 'nonzero_basis_words': len(poly),
                         'exact_mass': str(norm)})
    # Independent direct ternary-table second moment for all lengths <=5.
    for n in range(1, min(maximum, 5) + 1):
        b = np.array(list(product((-1., 0., 1.), repeat=n)))
        probability = np.prod(np.where(b == 0, .5, .25), axis=1)
        values = gate(b, [float(x) for x in eta], maximum)
        assert abs(float(probability @ (values**2)) - mass(n, [float(x) for x in eta], maximum)) < 1e-14
    magnitude = Q(1, 16)
    @lru_cache(None)
    def augmented(n):
        if n == 0:
            return {(): Q(1)}
        result = {('B',) + word: coef for word, coef in augmented(n-1).items()}
        for j in range(n-1):
            for word, coef in augmented(n-j-2).items():
                for right, factor in [('K', Q(2**j)*eta[j]), ('D', -Q(2**(j+1))*eta[j])]:
                    key = ('K',) + ('B',)*j + (right,) + word
                    assert key not in result
                    result[key] = factor*coef
        if n >= 2:
            for word, coef in augmented(n-2).items():
                key = ('K', 'M') + word
                assert key not in result
                result[key] = magnitude*coef
        return result
    recurrence = [Q(1)]
    for n in range(1, maximum+1):
        value = recurrence[-1]/2 + sum((Q(2**(j+1))*eta[j]**2*recurrence[n-j-2] for j in range(n-1)), Q(0))
        if n >= 2:
            value += magnitude**2*recurrence[n-2]
        recurrence.append(value)
        norm = Q(0)
        for word, coefficient in augmented(n).items():
            nonb = [i for i, letter in enumerate(word) if letter != 'B']
            moment, outside = Q(1), word.count('B')
            for left, right in zip(nonb[::2], nonb[1::2]):
                assert word[left] == 'K'
                j = right-left-1
                outside -= j
                if word[right] == 'M':
                    assert j == 0
                    moment *= magnitude
                else:
                    moment *= eta[j]*(Q(-1, 2) if word[right] == 'D' else 1)
            moment *= Q(1, 2)**outside
            squared_norm = Q(1, 2)**word.count('B') * Q(1, 4)**word.count('D')
            assert coefficient == moment/squared_norm
            norm += coefficient**2*squared_norm
        assert norm == value
        rows.append({'n': n, 'gap': maximum, 'magnitude_feature': True,
                     'nonzero_basis_words': len(augmented(n)), 'exact_mass': str(norm)})
    return rows


def m_value(first_coordinate, rho):
    return (2 / math.pi) * np.arcsin(np.clip(rho * first_coordinate / math.sqrt(2), -1, 1))


def conditional_sum(rng, count, large):
    """Exact rejection sampler for |G1|+|G2| conditioned above/below fixed2."""
    result = np.empty(count)
    filled = 0
    while filled < count:
        candidates = np.abs(rng.normal(size=(2*(count-filled)+64, 2))).sum(axis=1)
        accepted = candidates[candidates >= 2] if large else candidates[candidates < 2]
        take = min(len(accepted), count-filled)
        result[filled:filled+take] = accepted[:take]
        filled += take
    return result


def coefficients(p, d, rho, index):
    n, batch, gap = p['coefficient_examples'], p['batch_size'], p['maximum_gap']
    rng = np.random.default_rng(p['coefficient_seed'] + 1009 * index)
    sums = np.zeros(gap + 1)
    probability = 1-math.erf(1.)**2
    normalizer = math.sqrt(.5*probability*(1-probability))
    magnitude_sum = 0.
    for start in range(0, n, batch):
        size = min(batch, n - start)
        z = rng.normal(size=(size, d))
        e = np.sign(z[:, 0]) if rho == 1 else erf(rho * z[:, 0] / math.sqrt(2 * (1-rho*rho)))
        state = z / np.linalg.norm(z, axis=1, keepdims=True)
        small = conditional_sum(rng, size, False)
        large = conditional_sum(rng, size, True)
        coordinate = rho*state[:, 0]
        denominator = 2*np.sqrt(np.maximum(1-coordinate**2, np.finfo(float).tiny))
        positive = normalizer*e*(erf(coordinate*large/denominator)-erf(coordinate*small/denominator))
        assert positive.min() >= -1e-14
        magnitude_sum += positive.sum()
        for j in range(gap + 1):
            sums[j] += np.sum(e * m_value(state[:, 0], rho))
            if j < gap:
                common_sign = rng.choice((-1., 1.), size=size)
                value = common_sign[:, None] * np.abs(rng.normal(size=(size, 2)))
                keys = rng.normal(size=(size, 2, d))
                keys[:, :, 0] = rho * value + math.sqrt(1-rho*rho) * keys[:, :, 0]
                winner = np.argmax(np.einsum('ncd,nd->nc', keys, state), axis=1)
                state = keys[np.arange(size), winner]
                state /= np.linalg.norm(state, axis=1, keepdims=True)
    mean = sums / n
    half = float(Q(p['normalized_hoeffding_halfwidth']))
    records = []
    for j, observed in enumerate(mean):
        raw = observed * 2.**(-j)
        cap = .5 * 4.**(-j)
        lower, upper = max(-cap, raw - half * 2.**(-j)), min(cap, raw + half * 2.**(-j))
        if j < 2:
            lower = max(0., lower)
        if rho == 0:
            assert raw == 0
            lower = upper = 0.
        inconsistent = lower > upper
        if inconsistent:
            lower, upper = (0. if j < 2 else -cap), cap
        point = min(upper, max(lower, raw))
        records.append({'j': j, 'raw_estimate': float(raw), 'normalized_raw_mean': float(observed),
                        'interval': [float(lower), float(upper)], 'projected_estimate': float(point),
                        'analytic_absolute_bound': cap, 'interval_inconsistent_with_analytic_bound': inconsistent})
    mag = float(magnitude_sum/n)
    mag_interval = [max(0., mag-2*normalizer*half), min(1., mag+2*normalizer*half)]
    if rho == 0:
        mag_interval = [0., 0.]
    return {'d': d, 'rho': rho, 'coefficients': records,
            'magnitude': {'threshold': 2, 'tail_probability': probability,
                          'normalizer': normalizer, 'estimate': mag, 'interval': mag_interval,
                          'integrand_range': [0., 2*normalizer]}}


