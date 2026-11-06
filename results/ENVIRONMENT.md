# Environment and changes to the authors' code

## Machine and software
| | |
|---|---|
| Hardware | Apple M6 Mac mini, 32 GB RAM, Apple-GPU (PyTorch `mps` backend) |
| OS | macOS 27.0 |
| GPT2SP code | github.com/awsm-research/gpt2sp, commit **1117294** (merge of Tawosi's MAE fix, PR #2) |
| Python | 3.11.16 in `.venv` (uv); torch 2.15.0, transformers 4.57.6, tokenizers 0.22.2, pandas 3.1.0, numpy 2.4.6 |
| Pre-trained model | `gpt2` (117M) from the Hugging Face Hub; released GPT2SP models `MickyMike/0-GPT2SP-<project>` |
| Deep-SE | SOLAR-group/AgileEffortEstimation commit 291e22a, Python 2.7.18 + Theano 0.9.0 + Keras 1.0.6 in Docker (`linux/amd64` under Rosetta); see `replication/deepse/PATCHES.md` |
| Statistics | R 4.6.1, ScottKnottESD 2.0.3, effsize |
| TAWOS database | MySQL 26.10 (Homebrew), database `tawos` loaded from the official TAWOS v1.1 dump |

`transformers` must stay below 5.0: `GPT2SP.py` calls `self.init_weights()` in `__init__`, which breaks
`from_pretrained` in transformers 5.x.

## Changes to the GPT2SP code (all mechanical; listed in the script docstrings too)
`model_training_notebook.ipynb` was ported to `replication/train_gpt2sp.py`. Same model class (`GPT2SP.py`, unmodified),
tokenizer, title-only input, sequence length 20, 20 epochs, lr 5e-4, dynamic batch size (0.3 within / 0.4 cross),
L1 loss, linear schedule, gradient clipping 1.0, positional 60/20/20 split, model selection by lowest validation loss (`<=`, ties go to the later epoch).

1. Device: `cuda` -> `mps` -> `cpu` instead of the hard-coded `.cuda()`.
2. Large notebook batches (up to 968 issues) are run as micro-batches of 128 with exact gradient accumulation:
   each micro-batch loss is weighted by its share of the batch, so the gradient and the single optimizer step per
   notebook batch are identical to the original.
3. `pd.concat` instead of the removed `DataFrame.append`; `torch.optim.AdamW` instead of the removed `transformers.AdamW`.
4. Seeds fixed (`random`, `numpy`, `torch`); runs use seeds 1, 2, 3. MPS kernels are not bit-deterministic, so reruns can differ slightly.
5. Per-issue test predictions of the selected epoch are saved, and the MAE is recomputed from them (`replication/metrics.py`).
   Each run also reports the inspection-notebook formula on the same predictions.
6. Models are kept in memory rather than saved every epoch (`--save-model` saves the selected one); TensorBoard logging is dropped.
7. Labels: kept as the notebook's `LongTensor` for training, scored against the true float labels (only matters for TAWOS half points).

`replication/check_published_checkpoints.py` uses the inspection notebook's evaluation, with one change: its data path
`sp_dataset/marked_training_data/` does not exist in the repository, so it uses `sp_dataset/marked_data/`.
