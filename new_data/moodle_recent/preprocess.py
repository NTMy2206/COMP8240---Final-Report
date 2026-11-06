"""Newly constructed dataset, step 2: apply the benchmark's fields and filtering rules.

Input: raw/page_*.json written by collect.py. Filters, in order (each step is logged to funnel.csv):
  1. raw API results              every MDL issue with a story point, created after 2016-08-08
  2. Done + resolution Fixed/Done  status category Done (Moodle's "Closed") and resolution Fixed or Done,
                                   i.e. completed work whose estimate is final (benchmark: resolved issues)
  3. story point > 0              drops 0 and the -1 placeholder
  4. integer story point          the benchmark only has integer labels
  5. not an Epic                  epics are containers, not estimable items
  6. non-empty title              GPT2SP reads the title only
  7. not a sprint CLR container   Moodle HQ creates one issue per developer and sprint that carries the
                                  developer's Component Lead Review (CLR) velocity, titled e.g.
                                  "Sprint 3.1 2026: CLR issues by <name>" or "CLR list of issues by <name>".
                                  Its story point is a sprint budget, not the size of one piece of work.

Outputs:
  moodle_recent_full.csv           all kept issues with their metadata, oldest first
  marked_data/moodle_recent.csv    GPT2SP format: issuekey,title,description,storypoint,split_mark
                                   (positional 60/20/20 split by creation time, as in the benchmark)
  funnel.csv, stats.json

Usage: python new_data/moodle_recent/preprocess.py
"""
import glob
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CLR_CONTAINER = r"CLR issues by|^\s*CLR list of|^\s*CLONE - .*CLR issues by"


def load_raw():
    rows = []
    for f in sorted(glob.glob(str(HERE / "raw" / "page_*.json"))):
        for i in json.load(open(f))["issues"]:
            F = i["fields"]
            rows.append({
                "issuekey": i["key"], "title": (F.get("summary") or "").strip(),
                "description": F.get("description") or "", "storypoint": F.get("customfield_10043"),
                "created": F.get("created"), "resolutiondate": F.get("resolutiondate"),
                "updated": F.get("updated"), "status": F["status"]["name"],
                "status_category": F["status"]["statusCategory"]["name"],
                "resolution": (F.get("resolution") or {}).get("name"),
                "issuetype": F["issuetype"]["name"], "priority": (F.get("priority") or {}).get("name"),
                "components": ";".join(c["name"] for c in F.get("components") or [])})
    df = pd.DataFrame(rows).drop_duplicates("issuekey")
    df["created"] = pd.to_datetime(df["created"], utc=True)
    return df.sort_values(["created", "issuekey"]).reset_index(drop=True)


def sp_summary(s):
    s = s.astype(float)
    return {"n": int(len(s)), "mean": round(float(s.mean()), 2), "median": float(s.median()),
            "max": float(s.max())}


def main():
    df = load_raw()
    funnel = [("raw API results", len(df))]
    steps = [
        ("Done + resolution Fixed/Done",
         lambda d: (d.status_category == "Done") & d.resolution.isin(["Fixed", "Done"])),
        ("story point > 0", lambda d: d.storypoint.astype(float) > 0),
        ("integer story point", lambda d: d.storypoint.astype(float) % 1 == 0),
        ("not an Epic", lambda d: d.issuetype != "Epic"),
        ("non-empty title", lambda d: d.title.str.len() > 0),
        ("not a sprint CLR container", lambda d: ~d.title.str.contains(CLR_CONTAINER, case=False, regex=True)),
    ]
    removed_clr = None
    for name, cond in steps:
        keep = cond(df)
        if name.startswith("not a sprint CLR"):
            removed_clr = df[~keep]
        df = df[keep].reset_index(drop=True)
        funnel.append((name, len(df)))
    df["storypoint"] = df["storypoint"].astype(float).astype(int)

    pd.DataFrame(funnel, columns=["step", "issues"]).to_csv(HERE / "funnel.csv", index=False)
    removed_clr[["issuekey", "title", "storypoint"]].to_csv(HERE / "removed_clr_containers.csv", index=False)
    df.to_csv(HERE / "moodle_recent_full.csv", index=False)
    n = len(df)
    a, b = int(n * 0.6), int(n * 0.8)
    df["split_mark"] = ["train"] * a + ["val"] * (b - a) + ["test"] * (n - b)
    (HERE / "marked_data").mkdir(exist_ok=True)
    df[["issuekey", "title", "description", "storypoint", "split_mark"]].to_csv(
        HERE / "marked_data" / "moodle_recent.csv", index=False)

    bench = pd.read_csv(ROOT / "sp_dataset" / "marked_data" / "moodle.csv")
    manifest = json.loads((HERE / "raw" / "MANIFEST.json").read_text())
    stats = {
        "collected_at": manifest["collected_at"],
        "funnel": dict(funnel),
        "created_range": [str(df.created.min()), str(df.created.max())],
        "split": {"train": a, "val": b - a, "test": n - b,
                  "test_created_from": str(df.created.iloc[b])},
        "storypoints_benchmark_2016": sp_summary(bench["storypoint"]),
        "storypoints_recent": sp_summary(df["storypoint"]),
        "issuetype": df.issuetype.value_counts().to_dict(),
        "empty_description_frac": round(float((df.description.str.strip() == "").mean()), 3),
        "issues_per_year": df.created.dt.year.value_counts().sort_index().to_dict(),
        "mean_sp_per_year": df.groupby(df.created.dt.year).storypoint.mean().round(2).to_dict(),
    }
    (HERE / "stats.json").write_text(json.dumps(stats, indent=2, default=str) + "\n")
    print(pd.DataFrame(funnel, columns=["step", "issues"]).to_string(index=False))
    print("\nStory points, benchmark 2016 vs recent:", stats["storypoints_benchmark_2016"], "vs",
          stats["storypoints_recent"])
    print("split:", stats["split"])


if __name__ == "__main__":
    main()
