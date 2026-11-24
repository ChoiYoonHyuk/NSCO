"""Print Appendix L paired frame-error means and pointwise Student-t half-widths."""
from pathlib import Path
import json
from math import sqrt
from statistics import mean, stdev
from scipy.stats import t

root = Path(__file__).resolve().parent

def show(name, subset, key):
    values = [row['frame_error'] for row in subset if row[key] == name]
    if not values:
        raise ValueError(f'Missing estimator: {name}')
    radius = t.ppf(.975, len(values)-1) * stdev(values) / sqrt(len(values))
    print(f'{name}: {mean(values):.5f} +/- {radius:.5f} ({len(values)} runs)')

for name, estimators, key in [
    ('first_sign', ('corrected', 'orthogonal'), 'estimator'),
    ('fractional_weighting', ('certificate', 'fractional_control'), 'method'),
]:
    rows = json.loads((root/'results'/name/'raw_results.json').read_text())
    budget = max(row['n'] for row in rows)
    subset = [row for row in rows if row['n'] == budget and row['rho'] == 1.0]
    if key == 'method':
        subset = [row for row in subset if row['readout'] == 'majority_product' and row['k'] == 9]
    print(f'{name}, rho=1, n={budget}')
    for estimator in estimators:
        show(estimator, subset, key)
