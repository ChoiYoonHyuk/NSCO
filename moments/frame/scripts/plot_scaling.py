"""Compute both readout slopes and draw Figure 3 from the generated seed records."""
from pathlib import Path
import hashlib
import json
import math
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T95 = 2.200985160082949
COLORS = {"parity": "#9c2f38", "majority_product": "#1f6596",
          "odd_majority": "#648434", "random_balanced": "#8a5a9b"}
NAMES = {"parity": "Parity", "majority_product": "Majority product",
         "odd_majority": "Odd majority", "random_balanced": "Balanced random"}
MARKERS = {"parity": "o", "majority_product": "s",
           "odd_majority": "D", "random_balanced": "P"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def stats(values):
    assert len(values) == 12
    mean = statistics.mean(values)
    half = T95 * statistics.stdev(values) / math.sqrt(12)
    return {"mean": mean, "ci95_half_width": half,
            "seed_values": list(values)}


def main():
    p = read("scaling_settings.json")
    raw = read("results/frame_scaling/raw_results.json")
    manifest = read("results/frame_scaling/run_manifest.json")
    assert manifest["output_sha256"]["raw_results.json"] == sha(ROOT / "results/frame_scaling/raw_results.json")
    conditions = sorted({r["condition"] for r in raw if r["annotation_probability"] == 0})
    crossings = []
    for condition in conditions:
        for seed in p["seeds"]:
            rows = sorted((r for r in raw if r["condition"] == condition and r["seed"] == seed),
                          key=lambda r: r["n"])
            assert [r["n"] for r in rows] == p["sample_sizes"]
            hit = next((j for j, r in enumerate(rows) if r["frame_error"] <= .35), None)
            item = {k: rows[0][k] for k in ("condition", "seed", "family", "depth", "beta", "kappa1")}
            if hit is None:
                item.update(censoring="right", n_lower=rows[-1]["n"], n_upper=None,
                            n_grid=None, n_interpolated=None)
            elif hit == 0:
                item.update(censoring="left", n_lower=None, n_upper=rows[0]["n"],
                            n_grid=rows[0]["n"], n_interpolated=None)
            else:
                lo, hi = rows[hit - 1], rows[hit]
                fraction = ((math.log(.35) - math.log(lo["frame_error"])) /
                            (math.log(hi["frame_error"]) - math.log(lo["frame_error"])))
                value = math.exp(math.log(lo["n"]) + fraction * math.log(hi["n"] / lo["n"]))
                item.update(censoring="none", n_lower=lo["n"], n_upper=hi["n"],
                            n_grid=hi["n"], n_interpolated=value)
            crossings.append(item)
    assert len(crossings) == 168
    censored = [r for r in crossings if r["censoring"] != "none"]
    assert len(censored) == 1 and censored[0]["condition"] == "parity_k9" and censored[0]["seed"] == 10101
    groups = []
    for condition in conditions:
        rows = [r for r in crossings if r["condition"] == condition]
        groups.append({**{k: rows[0][k] for k in ("condition", "family", "beta", "kappa1")},
                       "rows": rows, "complete": all(r["censoring"] == "none" for r in rows)})
    complete = [g for g in groups if g["family"] != "first_value" and g["complete"]]
    assert len(complete) == 12
    bootstrap = np.random.default_rng(20261021).integers(0, 12, size=(10000, 12))
    fits = {}
    for quantity in ("kappa1", "beta"):
        x = np.array([-math.log2(g[quantity]) for g in complete])
        y = np.array([[math.log2(r["n_interpolated"]) for r in g["rows"]] for g in complete])
        xc = x - x.mean()
        slope = float(xc @ y.mean(1) / (xc @ xc))
        bs = xc @ y[:, bootstrap].mean(2) / (xc @ xc)
        fit = {"slope": slope, "intercept": float(y.mean() - slope * x.mean()),
               "paired_seed_bootstrap_ci95": np.quantile(bs, [.025, .975]).tolist(),
               "complete_conditions": [g["condition"] for g in complete],
               "excluded_censored_conditions": ["parity_k9"]}
        fits[quantity] = fit

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.35, 2.65), layout="constrained")
    for group in groups:
        if group["family"] == "first_value":
            continue
        x = -math.log2(group["kappa1"])
        color, marker = COLORS[group["family"]], MARKERS[group["family"]]
        for j, row in enumerate(group["rows"]):
            observed = row["censoring"] == "none"
            y = math.log2(row["n_interpolated"] if observed else row["n_lower"] or row["n_upper"])
            ax.scatter(x + (j - 5.5) * .007, y, color=color,
                       marker=marker if observed else "^", s=8 if observed else 35,
                       alpha=.28 if observed else 1, linewidths=0, zorder=2)
        if group["complete"]:
            st = stats([math.log2(r["n_interpolated"]) for r in group["rows"]])
            ax.errorbar(x, st["mean"], yerr=st["ci95_half_width"], fmt=marker,
                        color=color, ms=4, capsize=2, lw=.8, zorder=3)
    fit = fits["kappa1"]
    xx = np.array(ax.get_xlim())
    ax.plot(xx, fit["intercept"] + fit["slope"] * xx, "--", c=".48", lw=.9, zorder=1)
    baseline = [math.log2(r["n_interpolated"]) for r in crossings if r["family"] == "first_value"]
    ax.axhline(statistics.mean(baseline), c=".2", ls=":", lw=.8)
    lo, hi = fit["paired_seed_bootstrap_ci95"]
    ax.text(.025, .97, f"Slope {fit['slope']:.2f}\n95% run interval [{lo:.2f}, {hi:.2f}]",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.set_xlabel(r"$-\log_2\kappa_1$")
    ax.set_ylabel(r"$\log_2 n$ at frame error $0.35$")
    ax.grid(alpha=.18)
    handles = [Line2D([], [], color=COLORS[k], marker=MARKERS[k], ls="", label=NAMES[k], ms=4)
               for k in COLORS]
    handles.append(Line2D([], [], c=".2", ls=":", label="First-value baseline", lw=.8))
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7)
    (ROOT / "figures").mkdir(exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(ROOT / f"figures/frame_recovery.{ext}", dpi=200,
                    metadata={"Title": "Local Fourier mass and sample cost", "Creator": "Matplotlib"})
    plt.close(fig)

    (ROOT / "results/frame_scaling/slopes.json").write_text(json.dumps(fits, indent=2), encoding="utf8")
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
