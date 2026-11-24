# Moment estimators and finite-depth prediction

Run these commands from this directory. Each study generates its own synthetic inputs. Python dependencies are NumPy, Matplotlib 3.8 or later, ReportLab, and pypdf. The first/last-position SGD control also uses PyTorch.

| Command | Manuscript result |
| --- | --- |
| `python reproduce.py frame` | RQ1, Figure 3(a) and Appendix L. Thirteen readouts, with the same twelve fully observed conditions in both regressions. |
| `python reproduce.py fixed-budget` | Appendix L. Six independent frames at the fixed budget of 262,144 labels. |
| `python reproduce.py depth` | RQ1, Figure 3(b), RQ3, Table 2, and Appendix P. Trains both supervision arms at depths 3, 6, and 9, writes the depth figure and paired statistics, and prints Table 2. |
| `python reproduce.py select` | RQ1, Figure 3(c) and Appendix T.1. Best-position selection and fixed-position control. |
| `python reproduce.py controls` | Appendix P. Matched-stability moments, first versus last position, and zero versus random initialization. |
| `python reproduce.py collision` | Appendix P. Shared-input partial-path collisions across orthogonal frames. |

Results are written under the corresponding study directories. Use a fresh extraction for a new run of a study that already has a completed protocol or seed file. For the depth PDF, set `TIMES_FONT_DIR` to a directory with `times.ttf` and `timesbd.ttf` to use the original font.
