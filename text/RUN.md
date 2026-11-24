# Document and token experiments

Run the commands from this directory on Linux. The scorers and encoders use CUDA. Install a CUDA-compatible PyTorch, then install the shared dependencies.

```sh
python -m pip install numpy scipy transformers==4.48.0 huggingface_hub sentencepiece pyarrow munch==4.0.0 bidict==0.23.1 pyyaml
export PYTHONHASHSEED=0
```

| Directory | Experiment |
| --- | --- |
| `retrieval/` | Frozen E5 and learned two-stage scorers on MuSiQue, HotpotQA and 2WikiMultiHopQA (Appendix N) |
| `controlled/`, `bridge/` | Controlled MuSiQue signs, support supervision, one- and eight-draw comparison (Appendix R) |
| `musique_transfer/`, `hover/` | Frozen-scorer transfer, overlap audit and paired intervals (Appendix R) |
| `token/` | Native token task, sign readouts, shifted labels and frozen length/depth transfer (Appendix S) |

## Document data and representations

Download the official [MuSiQue release](https://github.com/stonybrooknlp/musique) to the compute server as `retrieval/data/musique_official.zip`. The archive SHA-256 is in `retrieval/data/manifest.json`. The other dataset preparation scripts below fetch HotpotQA and 2WikiMultiHopQA from their specified releases.

```sh
unzip -j retrieval/data/musique_official.zip 'data/musique_ans_v1.0_train.jsonl' 'data/musique_ans_v1.0_dev.jsonl' -d retrieval/data
python retrieval/scripts/initialize_cache_inputs.py
python retrieval/scripts/encode_passages.py
for split in train dev; do python retrieval/scripts/encode_context.py --dataset musique --split "$split"; done
python retrieval/scripts/prepare_splits.py
python retrieval/scripts/prepare_wiki_reference.py
python retrieval/scripts/prepare_hotpot.py
python retrieval/scripts/prepare_wiki_transfer.py
python retrieval/scripts/encode_hotpot_context.py --dataset hotpot --split dev
python retrieval/scripts/encode_wiki_context.py --dataset 2wiki --split dev
```

## Native document scoring

The trainer reads the final learning rates and residual caps from `retrieval/selected_settings.json`. The evaluation writes terminal-support accuracy, first-support accuracy, ordered-path accuracy and set metrics separately. For the context scorer with unit first-support loss, the original −0.03 and −4.30 percentage-point contrasts correspond to terminal-support accuracy (`terminal_r1`). Appendix N describes those numbers as first-support accuracy. The corresponding first-support changes (`first_r1`) are +10.08 and +7.60 points, respectively.

```sh
python retrieval/scripts/train_scorers.py
for target in lockbox musique hotpot 2wiki; do python retrieval/scripts/evaluate_scorers.py --target "$target"; done
python retrieval/scripts/summarize_scores.py
python retrieval/scripts/bootstrap_scores.py --run
```

## Controlled document labels and transfer

Run the MuSiQue cache preparation above first. The BGE snapshot path must end in its revision hash. The HoVer candidate builder uses the full official HotpotQA distractor JSON. If the native scoring preparation used its Parquet fallback, obtain that official JSON separately and set `HOTPOT_DEV` to its path.

```sh
export MUSIQUE_SOURCE="$PWD/retrieval/data"
export HOVER_E5="$PWD/retrieval/.hf_cache/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
export MUSIQUE_BGE="$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("BAAI/bge-reranker-v2-m3", revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", cache_dir="retrieval/.hf_cache/hub"))')"
export HOVER_BGE="$MUSIQUE_BGE"
export HOTPOT_DEV="$PWD/retrieval/data/hotpot/hotpot_dev_distractor_v1.json"
for mode in prepare run report; do python controlled/controlled_musique.py "$mode"; done
python bridge/prepare.py
python bridge/bridge.py fit
python bridge/bridge.py evaluate
python bridge/audit.py
python bridge/raw_pairs.py
python bridge/one_draw.py
python musique_transfer/holdout.py
python musique_transfer/review.py
python hover/export_weights.py
python hover/candidates.py
python hover/features.py
python hover/score.py
MUSIQUE_SPLIT="$PWD/bridge/split.private.json" MUSIQUE_TRAIN="$MUSIQUE_SOURCE/musique_ans_v1.0_train.jsonl" PREVIOUS_HOTPOT="$PWD/retrieval/data/hotpot/examples.jsonl" python hover/source_overlap.py
```

The paper's fully purged MuSiQue transfer cohort is empty. The transfer evaluation reports the sizes and results of the other declared cohorts.

## Token experiments

Initialize the pinned upstream generator and released native checkpoint with Git LFS. The remaining estimators are fit by the included scripts. Run in the indicated order to create the five frozen fits used by the length/depth evaluation.

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/chsanford/hop-induction-heads.git token/upstream
GIT_LFS_SKIP_SMUDGE=1 git -C token/upstream checkout 5735bca9545545774098994e21392b51b285bcc5
git -C token/upstream lfs pull --include="models/induction_hops_full_seq/width4-depth6/model_100000.pt"
python token/native_check.py
python token/sign_variant.py
python token/small_transformer.py
python token/audit_hop.py
python token/report.py
python token/shifted/fit_shifted.py
python token/shifted/fresh_check.py
python token/shifted/report.py
python token/transfer/evaluate.py token/shifted
```
