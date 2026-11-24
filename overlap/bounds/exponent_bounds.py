"""Reproduce the finite-filter Monte Carlo and analytic overlap bounds in Appendix O."""
import csv
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction as Q
from itertools import product
import math
from pathlib import Path

import numpy as np

OUTPUT = Path("results/overlap_bounds")
DIMENSIONS = (2, 4, 8)
OVERLAPS = (0.0, 0.25, 0.5, 1.0)
PROTOCOL = {"maximum_gap": 8, "coefficient_examples": 2**18, "batch_size": 32768,
            "coefficient_seed": 190723, "normalized_hoeffding_halfwidth": "9/2000"}
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid'))
from filters import coefficients

def exponent_upper_bound(root_lower, digits=4):
    denominator = 10**digits
    with localcontext() as context:
        context.prec = 90
        r = Decimal(root_lower.numerator) / Decimal(root_lower.denominator)
        numerator = int((-denominator * r.ln() / Decimal(2).ln()).to_integral_value(rounding=ROUND_CEILING))
    assert root_lower.numerator**denominator * 2**numerator >= root_lower.denominator**denominator
    return f"{Decimal(numerator) / denominator:.4f}"


def finite_filter_bound(row):
    weights = []
    for item in row["coefficients"]:
        lo, hi = map(lambda x: Q(str(x)), item["interval"])
        squared = Q(0) if lo <= 0 <= hi else min(lo**2, hi**2)
        weights.append(2**(item["j"] + 1) * squared)
    weights[0] += Q(str(row["magnitude"]["interval"][0]))**2
    with localcontext() as context:
        context.prec = 90
        ws = [Decimal(w.numerator) / Decimal(w.denominator) for w in weights]
        low, high = Decimal(".5"), Decimal(1)
        for _ in range(260):
            r = (low + high) / 2
            residual = r - Decimal(".5") - sum(w / r**(j+1) for j, w in enumerate(ws))
            if residual <= 0:
                low = r
            else:
                high = r
        lower = Q(low.quantize(Decimal("0.000000000001"), rounding=ROUND_FLOOR))
    assert lower - Q(1,2) - sum(w / lower**(j+1) for j, w in enumerate(weights)) <= 0
    return exponent_upper_bound(lower)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    half_width = float(Q(PROTOCOL["normalized_hoeffding_halfwidth"]))
    failure_bound = 240 * math.exp(-2 * PROTOCOL["coefficient_examples"] * half_width**2)
    table = OUTPUT / "exponent_bounds.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("dimension", "overlap", "finite_filter_exponent_upper"))
        for index, (d, rho) in enumerate(product(DIMENSIONS, OVERLAPS)):
            cell = coefficients(PROTOCOL, d, rho, index)
            bound = '1.0000' if rho == 0 else finite_filter_bound(cell)
            writer.writerow((d, rho, bound))
    print(f"{table}: 120 simultaneous coefficient means; failure probability bound = {failure_bound:.9f}")


if __name__ == "__main__":
    main()
