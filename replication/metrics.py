"""Error measures recomputed from raw per-issue predictions.

`mae` is the correct Mean Absolute Error (Eq. 1 of the paper): the mean over
all test issues.

`inspection_notebook_mae` reproduces the formula still used in
GPT2SP_inspection_notebook.ipynb (and in the training notebook before the
fix in commit 4d05140, Tawosi et al. 2022):

    total_distance / (len(predictions) * len(predictions[0]))

i.e. the sum of absolute errors divided by (number of batches x size of the
first batch). The last batch is usually smaller, so the denominator exceeds
the true number of issues and the MAE is too low.
"""
import math

import numpy as np
import pandas as pd


def abs_errors(actual, predicted):
    return np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))


def mae(actual, predicted):
    return float(np.mean(abs_errors(actual, predicted)))


def mdae(actual, predicted):
    return float(np.median(abs_errors(actual, predicted)))


def inspection_notebook_mae(actual, predicted, batch_size):
    """Buggy MAE: sum|e| / (n_batches * len(first batch))."""
    e = abs_errors(actual, predicted)
    n = len(e)
    n_batches = math.ceil(n / batch_size)
    first = min(batch_size, n)
    return float(e.sum() / (n_batches * first))


def notebook_batch_size(n_train_rows, ratio):
    """Dynamic batch size of the notebooks: int(len(train_text) * ratio)."""
    return int(n_train_rows * ratio)


def load_preds(path):
    df = pd.read_csv(path)
    return df["actual"].to_numpy(dtype=float), df["predicted"].to_numpy(dtype=float)


def summarise(actual, predicted, batch_size=None):
    out = {"n": len(actual), "mae": mae(actual, predicted), "mdae": mdae(actual, predicted)}
    if batch_size:
        out["mae_inspection_formula"] = inspection_notebook_mae(actual, predicted, batch_size)
    return out
