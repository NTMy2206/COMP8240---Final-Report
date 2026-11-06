"""Summarise the LLM-judge runs in results/llm_judge/raw.jsonl.

Outputs (results/llm_judge/):
  estimate.csv    the judge as an estimator: MAE per repetition vs GPT2SP / Median on the same items
  pointwise.csv   mean 1-5 score and share rated >= 3 ("plausible") per candidate and prompt wording
  pairwise.csv    GPT2SP vs Median: order consistency, preference, agreement with the recorded label
  summary.json    all headline numbers, including validity and reliability checks

Validity: does the judge score an estimate lower the further it is from the recorded story point
(Spearman between score and |candidate - recorded|)? Does it prefer, in the pairwise task, the
estimate that is closer to the recorded value?
Reliability: run-to-run agreement (Krippendorff's alpha over 3 repetitions), prompt sensitivity
(v1 vs v2 wording) and position bias (same item, A/B swapped).

Usage: python llm_judge/analyse.py
"""
import json
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "llm_judge"


def kalpha(wide, level):
    data = wide.T.to_numpy(dtype=float)
    if data.shape[0] < 2 or ((~np.isnan(data)).sum(axis=0) >= 2).sum() == 0:
        return float("nan")
    return float(krippendorff.alpha(reliability_data=data, level_of_measurement=level))


def main():
    raw = pd.DataFrame([json.loads(l) for l in (OUT / "raw.jsonl").open()])
    items = pd.read_csv(OUT / "items_with_candidates.csv").set_index("item_id")
    raw["score"] = raw.parsed.map(lambda p: p.get("score") if isinstance(p, dict) else None)
    raw["estimate"] = raw.parsed.map(lambda p: p.get("estimate") if isinstance(p, dict) else None)
    raw["choice"] = raw.parsed.map(lambda p: p.get("choice") if isinstance(p, dict) else None)
    S = {"model": sorted(raw.model.unique().tolist()), "calls": len(raw),
         "unparsed": int(((raw.task == "pointwise") & raw.score.isna()).sum()
                         + ((raw.task == "estimate") & raw.estimate.isna()).sum()
                         + ((raw.task == "pairwise") & ~raw.choice.isin(["A", "B"])).sum()),
         "cost_usd_list_price": round(float(raw.cost_usd.sum()), 2)}

    # --- estimator -----------------------------------------------------------------------------
    e = raw[raw.task == "estimate"].pivot(index="item_id", columns="rep", values="estimate").astype(float)
    rec = items.loc[e.index, "recorded"]
    rows = [{"estimator": f"LLM judge, rep {r}", "mae": (e[r] - rec).abs().mean()} for r in e.columns]
    rows.append({"estimator": "LLM judge, median of reps", "mae": (e.median(axis=1) - rec).abs().mean()})
    rows.append({"estimator": "GPT2SP seed 1 (raw)", "mae": (items.loc[e.index, "gpt2sp_raw"] - rec).abs().mean()})
    rows.append({"estimator": "GPT2SP seed 1 (rounded)", "mae": (items.loc[e.index, "gpt2sp"] - rec).abs().mean()})
    rows.append({"estimator": "Median baseline", "mae": (items.loc[e.index, "median"] - rec).abs().mean()})
    est = pd.DataFrame(rows)
    est.to_csv(OUT / "estimate.csv", index=False)
    S["estimate_mae"] = {r.estimator: round(r.mae, 3) for r in est.itertuples()}
    S["estimate_alpha_interval_reps"] = kalpha(e, "interval")
    S["estimate_exact_same_all_reps"] = float((e.nunique(axis=1) == 1).mean())
    S["estimate_spearman_with_recorded"] = float(spearmanr(e.median(axis=1), rec).statistic)
    S["estimate_distribution"] = e[1].value_counts().sort_index().to_dict()

    # --- pointwise -----------------------------------------------------------------------------
    p = raw[raw.task == "pointwise"].copy()
    p["score"] = p.score.astype(float)
    p["cand_value"] = [items.loc[i, c] for i, c in zip(p.item_id, p.cand)]
    p["abs_err"] = (p.cand_value - items.loc[p.item_id, "recorded"].values).abs()
    pw = (p.groupby(["variant", "cand"])
          .agg(n=("score", "size"), mean_score=("score", "mean"), plausible=("score", lambda s: (s >= 3).mean()))
          .reset_index())
    pw.to_csv(OUT / "pointwise.csv", index=False)
    S["pointwise"] = {f"{r.variant}/{r.cand}": [round(r.mean_score, 2), round(r.plausible, 2)] for r in pw.itertuples()}
    v1 = p[p.variant == "v1"]
    nonrec = v1[v1.cand != "recorded"]
    S["validity_spearman_score_vs_abs_error_v1"] = float(spearmanr(v1.score, v1.abs_err).statistic)
    S["validity_spearman_score_vs_abs_error_v1_excl_recorded"] = float(spearmanr(nonrec.score, nonrec.abs_err).statistic)
    w = v1.pivot_table(index=["item_id", "cand"], columns="rep", values="score")
    S["pointwise_alpha_ordinal_reps_v1"] = kalpha(w, "ordinal")
    S["pointwise_exact_same_all_reps_v1"] = float((w.nunique(axis=1) == 1).mean())
    r1 = p[p.rep == 1].pivot_table(index=["item_id", "cand"], columns="variant", values="score").dropna()
    S["prompt_sensitivity_spearman_v1_v2"] = float(spearmanr(r1.v1, r1.v2).statistic)
    S["prompt_sensitivity_exact_v1_v2"] = float((r1.v1 == r1.v2).mean())
    S["prompt_sensitivity_mean_v1_minus_v2"] = float((r1.v1 - r1.v2).mean())
    # does the judge rate the recorded value highest?
    m = v1.groupby(["item_id", "cand"]).score.mean().unstack()
    S["recorded_rated_at_least_as_high_as_gpt2sp"] = float((m.recorded >= m.gpt2sp).mean())
    S["recorded_rated_at_least_as_high_as_median"] = float((m.recorded >= m["median"]).mean())

    # --- pairwise ------------------------------------------------------------------------------
    q = raw[raw.task == "pairwise"].copy()
    # translate A/B into the estimator chosen
    q["chosen"] = np.where((q.cand == "gpt2sp_first") == (q.choice == "A"), "gpt2sp", "median")
    q.loc[~q.choice.isin(["A", "B"]), "chosen"] = None
    q["chose_first"] = q.choice == "A"
    t = q.pivot_table(index=["item_id", "rep"], columns="cand", values="chosen", aggfunc="first").dropna()
    consistent = t.gpt2sp_first == t.median_first
    S["pairwise_items"] = int(q.item_id.nunique())
    S["pairwise_order_consistency"] = float(consistent.mean())
    S["pairwise_first_position_rate"] = float(q.chose_first.mean())
    cons = t[consistent].gpt2sp_first
    S["pairwise_prefers_gpt2sp_when_consistent"] = float((cons == "gpt2sp").mean())
    # ground truth: which estimate is closer to the recorded label (ties excluded)
    it = items.loc[t.index.get_level_values(0)]
    dg = (it.gpt2sp - it.recorded).abs().values
    dm = (it["median"] - it.recorded).abs().values
    truth = pd.Series(np.where(dg < dm, "gpt2sp", np.where(dm < dg, "median", "tie")), index=t.index)
    ok = consistent & (truth != "tie")
    S["pairwise_closer_is_gpt2sp_share"] = float((truth[truth != "tie"] == "gpt2sp").mean())
    S["pairwise_agrees_with_recorded_when_consistent"] = float((t[ok].gpt2sp_first == truth[ok]).mean())
    S["pairwise_n_consistent_nontie"] = int(ok.sum())
    pd.DataFrame({"measure": list(k for k in S if k.startswith("pairwise")),
                  "value": [S[k] for k in S if k.startswith("pairwise")]}).to_csv(OUT / "pairwise.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(S, indent=2, default=float))
    print(json.dumps(S, indent=2, default=float))


if __name__ == "__main__":
    main()
