"""Mean and Median Effort baselines (Shepperd & MacDonell 2012), from the same splits as GPT2SP.

Each baseline predicts one constant for every test issue: the mean / median story point of the
TRAINING issues (oldest 60% of the training project) - the history that GPT2SP itself learns from.
Test issues are the same as for GPT2SP (newest 20% within-project, or the whole target project
cross-project). Per-issue predictions are written so absolute errors can go into Scott-Knott ESD.

Usage:
  python replication/naive_baselines.py --scenario within
  python replication/naive_baselines.py --scenario within --data-dir new_data/tawos/marked_data \
      --projects ALOY APSTUD ... --tag tawos_within
  python replication/naive_baselines.py --scenario custom --train moodle --test moodle_recent \
      --data-dir new_data/moodle_recent/marked_data --tag moodle_2016_to_recent
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "replication"))
import replication.metrics as metrics  # noqa: E402
from replication.train_gpt2sp import (BENCHMARK, CROSS_REPO_PAIRS, DEFAULT_DATA_DIR, WITHIN_REPO_PAIRS,  # noqa: E402
                          find_csv, prepare_dataframe)


def baseline_preds(train_names, test_name, within, data_dirs):
    data = pd.concat([prepare_dataframe(find_csv(n, data_dirs)) for n in train_names], ignore_index=True)
    train = data.iloc[: int(len(data) * 0.6)]
    if within:
        test = data.iloc[int(len(data) * 0.8):]
    else:
        test = prepare_dataframe(find_csv(test_name, data_dirs))
    y = train["label"].to_numpy(dtype=float)
    return pd.DataFrame({"issuekey": test["issuekey"].to_numpy(),
                         "actual": test["label"].to_numpy(dtype=float),
                         "mean_pred": float(np.mean(y)), "median_pred": float(np.median(y))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["within", "within_repo", "cross_repo", "custom"], required=True)
    ap.add_argument("--projects", nargs="*")
    ap.add_argument("--train", nargs="*")
    ap.add_argument("--test")
    ap.add_argument("--custom-within", action="store_true")
    ap.add_argument("--data-dir", action="append", default=[])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    data_dirs = [ROOT / d if not os.path.isabs(d) else Path(d) for d in args.data_dir] + [DEFAULT_DATA_DIR]
    if args.scenario == "within":
        jobs = [([p], p, True) for p in (args.projects or BENCHMARK)]
    elif args.scenario == "within_repo":
        jobs = [([a], b, False) for a, b in WITHIN_REPO_PAIRS]
    elif args.scenario == "cross_repo":
        jobs = [([a], b, False) for a, b in CROSS_REPO_PAIRS]
    else:
        jobs = [(args.train, args.test or args.train[0], args.custom_within)]
    tag = args.tag or args.scenario
    out = ROOT / "results" / "baselines" / tag
    (out / "preds").mkdir(parents=True, exist_ok=True)
    rows = []
    for train_names, test_name, within in jobs:
        df = baseline_preds(train_names, test_name, within, data_dirs)
        stem = f"{'+'.join(train_names)}__{test_name}"
        df.to_csv(out / "preds" / f"{stem}_preds.csv", index=False)
        rows.append({"train": "+".join(train_names), "test": test_name, "n_test": len(df),
                     "mean_value": df["mean_pred"].iloc[0], "median_value": df["median_pred"].iloc[0],
                     "mean_mae": metrics.mae(df["actual"], df["mean_pred"]),
                     "median_mae": metrics.mae(df["actual"], df["median_pred"]),
                     "mean_mdae": metrics.mdae(df["actual"], df["mean_pred"]),
                     "median_mdae": metrics.mdae(df["actual"], df["median_pred"])})
    res = pd.DataFrame(rows)
    res.to_csv(out / "baselines.csv", index=False)
    print(res.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
