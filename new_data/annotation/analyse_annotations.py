"""Read the returned human-estimate workbooks and compute agreement and accuracy."""
import argparse
import itertools
import json
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / "results" / "human"
GPT2SP = ROOT / "results" / "gpt2sp" / "moodle_recent_within"
BASE = ROOT / "results" / "baselines" / "moodle_recent_within" / "preds" / "moodle_recent__moodle_recent_preds.csv"


def num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return np.nan


def read_workbook(path):
    wb = load_workbook(path, data_only=True)
    start = wb["Start here"]
    rows = [c.value for c in start["A"]]
    i = next(k for k, v in enumerate(rows) if v and "Type YES" in str(v))
    consent = str(rows[i + 1] or "").strip().upper().startswith("Y")
    est = pd.read_excel(path, sheet_name="Estimates")
    est = pd.DataFrame({"item_id": est["item"], "estimate": est["your_estimate_sp"].map(num),
                        "clarity": est["clarity_1to5"].map(num), "confidence": est["confidence_1to5"].map(num),
                        "comment": est["comment"].fillna("")})
    about = pd.read_excel(path, sheet_name="About you")["your answer"].tolist() + [None] * 4
    return consent, est, about[:4]


def alpha(wide, level):
    data = wide.T.to_numpy(dtype=float)  # raters x items
    keep = (~np.isnan(data)).sum(axis=0) >= 2
    return float(krippendorff.alpha(reliability_data=data[:, keep], level_of_measurement=level)) if keep.sum() > 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returned", type=Path, default=HERE / "returned")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    items = pd.read_csv(HERE / "items_shown.csv", usecols=["item_id", "issuekey", "part"])
    key = pd.read_csv(HERE / "answer_key.csv", usecols=["item_id", "storypoint"])
    files = sorted(args.returned.glob("*.xlsx"))
    if not files:
        raise SystemExit(f"no workbooks in {args.returned}")

    long, people = [], []
    for f in files:
        consent, est, about = read_workbook(f)
        if not consent:
            print(f"skipped (no consent): {f.name}")
            continue
        pid = f"P{len(people) + 1}"
        est.insert(0, "participant", pid)
        long.append(est)
        people.append({"participant": pid, "n_items": int(est.estimate.notna().sum()), "role": about[0],
                       "agile_years": about[1], "estimated_before": about[2], "moodle_familiarity": about[3]})
    h = pd.concat(long).merge(items, on="item_id")
    h = h[["participant", "item_id", "issuekey", "part", "estimate", "clarity", "confidence", "comment"]]
    out.mkdir(parents=True, exist_ok=True)
    h.to_csv(out / "human_estimates.csv", index=False)
    pd.DataFrame(people).to_csv(out / "participants.csv", index=False)

    wide = h.pivot(index="item_id", columns="participant", values="estimate")
    partA = wide.loc[wide.index <= items[items.part == "A"].item_id.max()]
    agree = [{"measure": f"krippendorff_alpha_{lvl}", "items": scope, "value": alpha(w, lvl)}
             for lvl in ("interval", "ordinal") for scope, w in (("all", wide), ("part A", partA))]
    for a, b in itertools.combinations(wide.columns, 2):
        both = wide[[a, b]].dropna()
        agree += [{"measure": "pairwise_exact_agreement", "items": f"{a}-{b} (n={len(both)})",
                   "value": float((both[a] == both[b]).mean())},
                  {"measure": "pairwise_mean_abs_diff", "items": f"{a}-{b} (n={len(both)})",
                   "value": float((both[a] - both[b]).abs().mean())},
                  {"measure": "pairwise_spearman", "items": f"{a}-{b} (n={len(both)})",
                   "value": float(spearmanr(both[a], both[b]).statistic)}]
    agree = pd.DataFrame(agree)
    agree.to_csv(out / "agreement.csv", index=False)

    # accuracy on identical items: an item counts once at least one person estimated it
    g = [pd.read_csv(GPT2SP / f"moodle_recent__moodle_recent__seed{s}_preds.csv")
         .rename(columns={"predicted": f"gpt2sp_s{s}"}).drop(columns="actual") for s in (1, 2, 3)]
    base = pd.read_csv(BASE).drop(columns="actual")
    df = items.merge(key, on="item_id").merge(base, on="issuekey")
    for x in g:
        df = df.merge(x, on="issuekey")
    df = df.merge(wide.rename_axis(columns=None), left_on="item_id", right_index=True, how="left")
    df["human_consensus"] = df[list(wide.columns)].median(axis=1)
    rows = []

    def add(name, pred, mask):
        e = (df.loc[mask, "storypoint"] - df.loc[mask, pred]).abs()
        rows.append({"estimator": name, "items": "covered" if mask is covered else name,
                     "n": int(mask.sum()), "mae": e.mean(), "mdae": e.median()})

    covered = df.human_consensus.notna()
    for p in wide.columns:
        m = df[p].notna()
        add(p, p, m)
    add("human consensus (median)", "human_consensus", covered)
    for s in (1, 2, 3):
        add(f"GPT2SP seed {s}", f"gpt2sp_s{s}", covered)
    add("Median baseline", "median_pred", covered)
    add("Mean baseline", "mean_pred", covered)
    acc = pd.DataFrame(rows)
    gp = acc[acc.estimator.str.startswith("GPT2SP")]
    acc = pd.concat([acc, pd.DataFrame([{"estimator": "GPT2SP (mean of 3 seeds)", "items": "covered",
                                         "n": int(covered.sum()), "mae": gp.mae.mean(), "mdae": gp.mdae.mean()}])])
    acc.to_csv(out / "accuracy.csv", index=False)

    clar = h.groupby("item_id").clarity.mean()
    df["clarity_mean"] = df.item_id.map(clar)
    df["err_human"] = (df.storypoint - df.human_consensus).abs()
    df["err_gpt2sp"] = (df.storypoint - df.gpt2sp_s1).abs()
    d = df.dropna(subset=["clarity_mean", "err_human"])
    summary = {
        "participants": len(people), "items_with_any_estimate": int(covered.sum()),
        "alpha_interval_all": agree.query("measure=='krippendorff_alpha_interval' and items=='all'").value.item(),
        "alpha_interval_partA": agree.query("measure=='krippendorff_alpha_interval' and items=='part A'").value.item(),
        "spearman_clarity_vs_human_error": float(spearmanr(d.clarity_mean, d.err_human).statistic),
        "spearman_clarity_vs_gpt2sp_error": float(spearmanr(d.clarity_mean, d.err_gpt2sp).statistic),
        "mae": {r.estimator: round(r.mae, 3) for r in acc.itertuples()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(people).to_string(index=False))
    print(agree.to_string(index=False))
    print(acc.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
