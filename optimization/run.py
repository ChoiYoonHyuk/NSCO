"""Run one Gaussian optimization study or regenerate the combined figures."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
JOBS = {
    "refinement": [
        ("refinement/scripts/averaged_sgd.py", "confirmation"),
        ("refinement/scripts/averaged_report.py",),
    ],
    "sample-split": [
        ("refinement/scripts/sample_split.py", "confirmation"),
        ("refinement/scripts/sample_split_report.py",),
    ],
    "six-step": [("six_step/gaussian.py",), ("six_step/report.py",)],
    "readout-swap": [("readout_swap/matched.py",), ("readout_swap/report.py",)],
    "matched-rates": [("six_step_rates/rate_recheck.py",)],
    "paired-depths": [("paired_depths/deep_adam.py",), ("paired_depths/summarize.py",)],
    "replication": [("replication/replicate.py",), ("replication/analyze.py",), ("replication/plot_curves.py",)],
    "shuffled-sign": [("shuffled_sign/control.py",), ("shuffled_sign/summarize.py",)],
    "common-path": [("common_path/readout.py",), ("common_path/summarize.py",)],
    "figures": [("figures/followup_comparison.py",), ("figures/reviewer_checks.py",), ("adam_label_depth.py",)],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=JOBS)
    args = parser.parse_args()
    for filename, *arguments in JOBS[args.study]:
        path = HERE / filename
        subprocess.run([sys.executable, str(path), *arguments], cwd=path.parent, check=True)
