"""Run the moment-estimator experiments from the current paper."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def run(folder, script, *args):
    subprocess.run([sys.executable, script, *map(str, args)], cwd=ROOT / folder, check=True)

def depth_study():
    folder = ROOT / "depth"
    from depth.study import P
    for seed in P["seeds"]:
        run("depth", "study.py", seed)
    run("depth", "report.py")
    run("depth", "paired.py")
    data = json.loads((folder / "paired_summary.json").read_text())["bracket"]["slopes"]
    print("\nTable 2: finite-depth label-cost slopes")
    print("Readout             Observed [95% paired CI]     A_k^2/kappa_1   1/beta_g")
    for family, name, benchmark in (("parity", "Parity", 1.), ("majority_product", "Majority product", math.log2(32 / 13) / 3)):
        mean, lo, hi = data[f"{family}/final"]
        extra = 2 * math.log2(9 / 3) / (9 - 3)
        print(f"{name:18} {mean:.2f} [{lo:.3f}, {hi:.3f}]             {benchmark + extra:.2f}          {benchmark:.2f}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["frame", "fixed-budget", "depth", "select", "controls", "collision"])
    args = parser.parse_args()
    if args.study == "frame":
        run("frame", "scripts/estimate_scaling.py", "--run")
        run("frame", "scripts/plot_scaling.py")
    elif args.study == "fixed-budget":
        run("frame", "scripts/fixed_budget.py")
    elif args.study == "depth":
        depth_study()
    elif args.study == "select":
        run("selection", "experiment.py")
    elif args.study == "collision":
        run("depth", "collision.py")
    elif args.study == "controls":
        for task in ("same_beta", "position", "sgd"):
            run("controls", "run_synthetic.py", task)
        run("controls", "analyze.py")

if __name__ == "__main__":
    main()
