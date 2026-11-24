# Gaussian Attention and Optimization

Run from this directory with Python 3.11+, NumPy, SciPy, Matplotlib, ReportLab, and CUDA-enabled PyTorch. The training inputs are synthetic. Start each new training study in a fresh copy of its directory.

| Manuscript experiment | Command |
| --- | --- |
| M.1. Averaged SGD and moment refinement | `python run.py refinement` |
| M.2. One-pass Gaussian SGD and Adam | `python run.py six-step` |
| M.2 and T.2. Six-step fixed-rate control | `python run.py matched-rates` |
| RQ2 and M.3. Equal stability, different first gradients | `python run.py readout-swap` |
| M.4. Paired nine- and twelve-step Adam | `python run.py paired-depths` |
| M.5. Independent twelve-step replication | `python run.py replication` |
| M.5. Cyclically shifted first-sign control | `python run.py shuffled-sign` |
| M.6. Exact paths under parity and majority-product training | `python run.py common-path` |
| N. Halfway freeze and sample-split SGD correction | `python run.py sample-split` |
| Mixed-panel appendix figures and T.2 equal-label plot from archived measurements | `python run.py figures` |

Each training script writes to its study's `results/` directory. The two refinement protocols share `refinement/` and write separate result trees. The fixed-choice JSON files contain the original development-selected settings. The mixed-panel plots use published summary measurements in `figures/inputs/`. To use the original plot font, set `PAPER_TIMES_DIR` (Matplotlib) or `TIMES_FONT_DIR` (ReportLab) to the directory containing the Times New Roman fonts.

For the T.2 figure and the M.4 paired depth contrast from new training runs, first run `matched-rates`, `paired-depths`, and `replication`, then run `python adam_label_depth.py --trained`. For the same calculation from the recorded teacher counts, run `python adam_label_depth.py`.
