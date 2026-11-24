# Noise Stability and the Statistical Query Cost of Outcome Supervision

## Files

| Directory | Experiments and run commands |
| --- | --- |
| [moments/](moments/RUN.md) | RQ1 and RQ3. Frame recovery, label counts, step selection, finite-depth prediction, and path collisions. |
| [optimization/](optimization/RUN.md) | RQ2 and the optimizer studies. First gradients, SGD refinement, Adam supervision, and depth replication. |
| [overlap/](overlap/RUN.md) | Key–value overlap. Exponent bounds, public-block simulation, filters, and calibration. |
| [text/](text/RUN.md) | Document retrieval and token lookup. Dataset preparation, training, and transfer evaluation. |

## Run the main experiments

Run from the extracted folder with Python 3.11 or later. The moment experiments generate synthetic inputs on CPU.

```sh
python -m pip install numpy scipy "matplotlib>=3.8" reportlab pypdf
python moments/reproduce.py frame        # RQ1: Figure 3(a)
python moments/reproduce.py depth        # RQ1: Figure 3(b); RQ3: Table 2
python moments/reproduce.py select       # RQ1: Figure 3(c)
```

For RQ2, install the CUDA build of PyTorch for your GPU driver in the same environment. Check that PyTorch can see the GPU, then run the matched-readout experiment.

```sh
python -c "import torch; assert torch.cuda.is_available()"
python optimization/run.py readout-swap  # RQ2
```

Each directory's run guide lists its appendix experiments, outputs, and dependencies. The text guide identifies the external datasets and checkpoints.
