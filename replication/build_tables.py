"""Build every results table from the raw per-issue predictions.

Inputs (all produced by the other scripts):
  results/gpt2sp/<tag>/runs.csv + *_preds.csv     GPT2SP runs (3 seeds)
  results/baselines/<tag>/preds/*_preds.csv       Mean / Median baselines
  results/published_checkpoints/                  authors' released models, re-scored
  results/deepse/preds/<project>_preds.csv        Deep-SE (Tawosi's package)
  results/reference/*.csv                         numbers copied from the papers
Outputs:
  results/tables/{within_project,cross_project,tawos_within,moodle_recent}.{csv,md}
  results/abs_errors_within.csv                   per-issue absolute errors for Scott-Knott ESD

All MAEs are recomputed here from predictions (replication/metrics.py). GPT2SP values are the mean
(and SD) over seeds of the per-seed MAE; for Scott-Knott, each issue's GPT2SP absolute error is
averaged over seeds, so the per-project mean of those errors equals the mean per-seed MAE.

Usage: python replication/build_tables.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "replication"))
import replication.metrics as metrics  # noqa: E402
from replication.train_gpt2sp import BENCHMARK, CROSS_REPO_PAIRS, WITHIN_REPO_PAIRS  # noqa: E402

RES = ROOT / "results"
TABLES = RES / "tables"
# Tawosi et al. (2022) GPT2SP replication report, Table 3a: pairs identical to the notebook's
TAWOSI_WITHIN_REPO = {("mesos", "usergrid"): 1.11, ("usergrid", "mesos"): 1.52,
                      ("appceleratorstudio", "aptanastudio"): 4.62,
                      ("appceleratorstudio", "titanium"): 3.28, ("aptanastudio", "titanium"): 4.09,
                      ("mule", "mulestudio"): 3.48, ("mulestudio", "mule"): 3.03}


def gpt2sp_runs(tag):
    p = RES / "gpt2sp" / tag / "runs.csv"
    if not p.exists():
        return pd.DataFrame()
    r = pd.read_csv(p).drop_duplicates(["train", "test", "seed"], keep="last")
    return r


def gpt2sp_summary(tag):
    r = gpt2sp_runs(tag)
    if r.empty:
        return pd.DataFrame(columns=["train", "test", "n_test", "seeds", "our_gpt2sp", "gpt2sp_sd",
                                     "gpt2sp_buggy_formula"])
    g = r.groupby(["train", "test"], sort=False)
    return pd.DataFrame({"n_test": g.n_test.first(), "seeds": g.seed.nunique(), "our_gpt2sp": g.mae.mean(),
                         "gpt2sp_sd": g.mae.std(ddof=1).fillna(0.0),
                         "gpt2sp_buggy_formula": g.mae_inspection_formula.mean()}).reset_index()


def gpt2sp_issue_errors(tag, train, test, subset=None):
    """Per-issue absolute error averaged over seeds (and the per-seed MAEs) for one pair."""
    files = sorted((RES / "gpt2sp" / tag).glob(f"{train}__{test}__seed*_preds.csv"))
    if not files:
        return None, []
    errs, maes = [], []
    for f in files:
        d = pd.read_csv(f)
        if subset is not None:
            d = d[d.issuekey.isin(subset)]
        e = pd.Series(metrics.abs_errors(d.actual, d.predicted), index=d.issuekey.to_numpy())
        errs.append(e)
        maes.append(e.mean())
    return pd.concat(errs, axis=1).mean(axis=1), maes


def baselines(tag):
    p = RES / "baselines" / tag / "baselines.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["train", "test", "mean_mae", "median_mae"])


def write(df, name, floatfmt=".2f"):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=floatfmt) + "\n")


def within_project():
    ref = pd.read_csv(RES / "reference" / "within_project_reference.csv")
    g = gpt2sp_summary("within").rename(columns={"test": "project"}).drop(columns="train")
    b = baselines("within").rename(columns={"test": "project"})[["project", "mean_mae", "median_mae"]]
    pc = RES / "published_checkpoints" / "published_checkpoints.csv"
    pub = (pd.read_csv(pc)[["project", "mae"]].rename(columns={"mae": "released_ckpt_mae"})
           if pc.exists() else pd.DataFrame(columns=["project", "released_ckpt_mae"]))
    dp = RES / "deepse" / "deepse_within.csv"
    ds = (pd.read_csv(dp)[["project", "mae"]].rename(columns={"mae": "our_deepse"})
          if dp.exists() else pd.DataFrame(columns=["project", "our_deepse"]))
    t = pd.DataFrame({"project": BENCHMARK})
    for other in (ref, g, pub, ds, b):
        t = t.merge(other, on="project", how="left")
    t = t[["project", "n_test", "paper_gpt2sp", "gpt2sp_buggy_formula", "tawosi_gpt2sp", "our_gpt2sp",
           "gpt2sp_sd", "released_ckpt_mae", "tawosi_deepse", "our_deepse", "tawosi_median", "median_mae",
           "mean_mae"]]
    med = t.drop(columns="project").median(numeric_only=True)
    t = pd.concat([t, pd.DataFrame([{"project": "MEDIAN", **med.to_dict()}])], ignore_index=True)
    write(t, "within_project")

    # per-issue absolute errors for Scott-Knott ESD, on the issues every method was tested on
    rows = []
    for p in BENCHMARK:
        bl = pd.read_csv(RES / "baselines" / "within" / "preds" / f"{p}__{p}_preds.csv")
        keys = set(bl.issuekey)
        dfile = RES / "deepse" / "preds" / f"{p}_preds.csv"
        dse = pd.read_csv(dfile) if dfile.exists() else None
        if dse is not None:
            keys &= set(dse.issuekey)
        g_err, _ = gpt2sp_issue_errors("within", p, p, keys)
        if g_err is None:
            continue
        keys &= set(g_err.index)
        bl = bl[bl.issuekey.isin(keys)]
        methods = {"GPT2SP": g_err.loc[bl.issuekey].to_numpy(),
                   "Mean": metrics.abs_errors(bl.actual, bl.mean_pred),
                   "Median": metrics.abs_errors(bl.actual, bl.median_pred)}
        if dse is not None:
            d = dse.set_index("issuekey").loc[bl.issuekey]
            methods["DeepSE"] = metrics.abs_errors(d.actual, d.predicted)
        for m, e in methods.items():
            rows.append(pd.DataFrame({"project": p, "issuekey": bl.issuekey.to_numpy(), "method": m,
                                      "abs_error": e}))
    if rows:
        pd.concat(rows).to_csv(RES / "abs_errors_within.csv", index=False)
    return t


def cross_project():
    out = []
    for scen, pairs in (("within_repo", WITHIN_REPO_PAIRS), ("cross_repo", CROSS_REPO_PAIRS)):
        g = gpt2sp_summary(scen)
        b = baselines(scen)
        for a, c in pairs:
            row = {"scenario": scen, "train": a, "test": c}
            gr = g[(g.train == a) & (g.test == c)]
            br = b[(b.train == a) & (b.test == c)]
            if len(gr):
                row.update(gr.iloc[0][["n_test", "seeds", "our_gpt2sp", "gpt2sp_sd", "gpt2sp_buggy_formula"]].to_dict())
            if len(br):
                row.update(br.iloc[0][["mean_mae", "median_mae"]].to_dict())
            row["tawosi_gpt2sp"] = TAWOSI_WITHIN_REPO.get((a, c), np.nan) if scen == "within_repo" else np.nan
            out.append(row)
    t = pd.DataFrame(out)
    if "our_gpt2sp" in t:
        t["gpt2sp_beats_median"] = t.our_gpt2sp < t.median_mae
    write(t, "cross_project")
    return t


def tawos_within():
    stats = pd.read_csv(ROOT / "new_data" / "tawos" / "stats.csv")
    ref = pd.read_csv(RES / "reference" / "tawos_reference.csv")
    g = gpt2sp_summary("tawos_within").rename(columns={"test": "key"}).drop(columns=["train", "n_test"])
    b = baselines("tawos_within").rename(columns={"test": "key"})[["key", "mean_mae", "median_mae"]]
    t = stats[["key", "repository", "in_gpt2sp_benchmark", "issues", "n_test"]]
    for other in (g, ref[["key", "tawosi_deepse", "tawosi_median"]], b):
        t = t.merge(other, on="key", how="left")
    t = t.rename(columns={"key": "project"})
    t["gpt2sp_beats_median"] = t.our_gpt2sp < t.median_mae
    t = t[["project", "repository", "in_gpt2sp_benchmark", "issues", "n_test", "seeds", "our_gpt2sp",
           "gpt2sp_sd", "tawosi_deepse", "mean_mae", "median_mae", "tawosi_median", "gpt2sp_beats_median"]]
    write(t, "tawos_within")
    return t


def moodle_recent():
    rows = []
    marked = pd.read_csv(ROOT / "new_data" / "moodle_recent" / "marked_data" / "moodle_recent.csv")
    test_keys = set(marked.loc[marked.split_mark == "test", "issuekey"])

    def add(setting, tag, train, evaluated_on, subset):
        err, maes = gpt2sp_issue_errors(tag, train, "moodle_recent", subset)
        bp = RES / "baselines" / tag / "preds" / f"{train}__moodle_recent_preds.csv"
        if err is None or not bp.exists():
            return
        bl = pd.read_csv(bp)
        if subset is not None:
            bl = bl[bl.issuekey.isin(subset)]
        rows.append({"setting": setting, "evaluated_on": evaluated_on, "n_test": len(bl),
                     "seeds": len(maes), "gpt2sp_mae": np.mean(maes),
                     "gpt2sp_sd": np.std(maes, ddof=1) if len(maes) > 1 else 0.0,
                     "mean_mae": metrics.mae(bl.actual, bl.mean_pred),
                     "median_mae": metrics.mae(bl.actual, bl.median_pred)})

    add("train recent -> test recent (newest 20%)", "moodle_recent_within", "moodle_recent",
        "recent test split", None)
    add("train 2016 benchmark -> test recent", "moodle_2016_to_recent", "moodle", "recent test split", test_keys)
    add("train 2016 benchmark -> test recent", "moodle_2016_to_recent", "moodle", "all recent", None)
    t = pd.DataFrame(rows)
    write(t, "moodle_recent")
    return t


def main():
    pd.set_option("display.width", 220)
    for name, fn in (("Within-project (16 benchmark projects)", within_project),
                     ("Cross-project", cross_project), ("TAWOS (23 projects)", tawos_within),
                     ("Recent Moodle", moodle_recent)):
        print(f"\n=== {name} ===")
        print(fn().round(2).to_string(index=False))


if __name__ == "__main__":
    main()
