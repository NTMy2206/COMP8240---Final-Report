# Deep-SE replication: changes to Tawosi et al.'s code

Upstream: https://github.com/SOLAR-group/AgileEffortEstimation, commit
`291e22a8bf812f5a6b3e241e0805159eab48b648` (2022-12-06). Only `Deep-SE/` plus the data it needs were vendored:

- `Deep-SE/` (all code, `NCE/models/*.json`, and the shipped pre-trained language models `NCE/bestModels/*.hdf5`, byte-identical to upstream)
- `datasets/Choet_Dataset/` (all 14 CSVs + README, unchanged)
- `datasets/Pretrain_Dataset/`: only the 8 repositories the Choet projects use (Apache, Appcelerator, Atlassian,
  Duraspace, Lsstcorp, Moodle, Mulesoft, Spring) + README, unchanged. DNNSoftware, Hyperledger, MongoDB and Sonatype were left out.

The full diff against upstream is in `upstream.patch`. Every changed line is tagged `# [PATCH replication]`.
No model, hyper-parameter, split, tokenisation, seed or metric code was changed.

| # | File | Change | Why |
|---|------|--------|-----|
| 1 | `data/run_script.py` | `datasetDict_` can come from `sys.argv[1]` (default unchanged: `'Porru_Dataset'`) | Dataset selection. Upstream says to edit the line by hand. |
| 2 | `data/run_script.py` | In the `Pretrain_Dataset` branch, skip a repo whose CSV is not present | Dataset selection. Only the 8 repos Choet needs are vendored. Each repo is processed independently, so the skipped repos don't affect the others. |
| 3 | `NCE/exp_lstm2v.py` | `mode` and `datasetDict` can come from `sys.argv[1:3]` (defaults unchanged: `lstm2vec`, `Choet_Dataset`) | Mode/dataset selection. Upstream says to edit by hand. |
| 4 | `NCE/lstm2vec.py` | Input path `'../data/' + data + '.pkl.gz'` changed to `'../data/files/' + ...` | Path bug. `data/preprocess_storypoint.py` writes to `data/files/`, and `classification/training.py` already reads from there. Without this fix the lstm2vec step can't find its input. |
| 5 | `classification/exp_script.py` | `exp` (results directory name) can come from `sys.argv[1]`; an optional project subset from `sys.argv[2:]` (defaults unchanged: `rq1_1`, all 14 Choet projects) | Lets each project run in its own container in parallel. The per-project `training.py` command line is identical to upstream. |
| 6 | `classification/create_model.py` (`SaveResult.write_ar`) | Also writes `<exp>/log/ar/pred_<saving>` with `actual,predicted` for every test issue, at full precision | Saves per-issue predictions. It is written at exactly the same moments as upstream's `ar/` and `raw_` files, so it holds the test predictions at the best-validation epoch. |

No Python-2 or Theano compatibility patches were needed. The pinned environment in the `Dockerfile`
(Python 2.7.18, Keras 1.0.6, Theano 0.9.0, numpy 1.16.6, pandas 0.24.2, scikit-learn 0.18.2, h5py 2.7.1, plus scipy 1.2.3,
six 1.16.0 and PyYAML 5.4.1 as unpinned transitive deps) ran the code unmodified.

## Environment choices (not code changes)
- `~/.keras/keras.json`: `backend: theano`, `floatx: float32` (upstream README says to set the backend to theano).
- `THEANO_FLAGS=floatX=float32,device=cpu,blas.ldflags=-lopenblas`. Upstream's commented example flag also uses `floatX=float32`.
- `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, `PYTHONHASHSEED=0`. Python 2 hash randomisation is already off by default, so this setting only makes that explicit.
- Image is `linux/amd64` (run under Rosetta on Apple silicon), because the Python 2.7 scipy, pandas and sklearn wheels exist only for x86_64.

## Upstream behaviours kept as-is (they affect the results)
- **Classification only uses the pre-trained *embedding*.** With `-pretrain finetune_lm`, `training.py` loads
  `NCE/bestModels/lstm2v_<Repo>_dim50.hdf5` and copies only the Embedding matrix (`prepare_data.load_weight`). The LSTM
  starts from random weights. The features produced by the NCE `lstm2vec` step (`NCE/data/lstm2v_*.pkl.gz`) are never read
  by `classification/`. We still run that step because the README's pipeline includes it.
- **Vocabulary is 500.** `exp_script.py` does not pass `-vocab`, so `training.py` uses the `prepare_data.arg_passing`
  default of 500 and drops all words with index >= 500.
- **The pre-training dictionary can't be reproduced.** `data/preprocess_pretrain.py` builds each repo dictionary from an
  unseeded `numpy.random.shuffle` 2/3 split. Upstream doesn't ship the dictionaries used to train the shipped language
  models. In a regenerated dictionary about 95–97% of the top-500 words are still in the top 500 (93% for Lsstcorp),
  but only about 5–16% keep the same index. So the shipped embedding rows mostly line up with different words than the
  ones they were trained on. We measured this by rebuilding the Atlassian and Lsstcorp dictionaries with two other seeds.
  This affects any re-run of this package, the student's earlier one included. We kept the upstream unseeded behaviour.
  The dictionaries and tokenised pickles from our run are kept in `Deep-SE/data/files/`.
- **Seed and baselines.** `training.py` seeds numpy with 1234. `baselines.do_baselines` then draws 1000 random-guess
  samples before the model is built, unless `<exp>/log/ar/<P>_rg_mae.txt` already exists. So `run_deepse.sh` deletes the
  experiment directory before training, to keep the RNG stream the same as a fresh upstream run.
- **Split.** The split is 60/20/20 by row order (`divide_data_sortdate.py`). Because of its `-1` offsets, the test set is
  rows `numTrain+numValid-1 .. end`, which is slightly more than 20%.
- **Model selection and stopping.** The test MAE reported is the one at the epoch with the best validation MAE.
  Training stops after the patience schedule ends (5 epochs, +5 after each LR halving, stop when patience > 30).
