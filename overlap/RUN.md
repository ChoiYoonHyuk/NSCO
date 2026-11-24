# Overlap experiments

Run commands from this directory with Python 3.10+, NumPy, SciPy and Matplotlib. All inputs are synthetic.

| Manuscript location | Directory | Experiment |
|---|---|---|
| Appendix L | `known/` | First-sign correction and fractional weighting |
| Appendix O | `bounds/` | Conditional public-block checks and finite-filter exponent bounds |
| Appendix O | `bounds/full_value/` | Two-sided exponent interval at dimension 2 and overlap 1 |
| Appendices P–Q | `grid/` | Four filters on the fixed-depth overlap grid |
| Appendices Q and T | `depth/` | Depth sweep, label thresholds and lower-budget sensitivity |
| Appendix Q | `calibration/` | Supplied, shortened and validation-selected overlap |

```sh
python -m pip install numpy scipy matplotlib

python known/scripts/first_sign.py --run
python known/scripts/fractional_weighting.py --run
python known/summarize.py

python bounds/public_blocks.py
python bounds/exponent_bounds.py
python bounds/full_value/integrate.py
python bounds/full_value/enclose.py

python grid/run_overlap.py confirmation
python grid/summarize_overlap.py
python grid/make_report.py

python depth/depth.py production
python depth/report.py
python depth/floor_sensitivity.py depth

python calibration/calibrate.py production
python calibration/audit_frames.py
python calibration/report.py
```

The full-value integration uses 33,554,432 replica pairs. Run `enclose.py` after integration to obtain the interval from the generated counts. For shorter simulations, substitute `smoke` for `confirmation` or `production`. Generate the full production outputs before running the reporting scripts. The finite-filter CSV gives one-sided bounds. The full-value interval at dimension 2 and overlap 1 comes from `bounds/full_value/interval.json`.

