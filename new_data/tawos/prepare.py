"""Existing new dataset: TAWOS (Tawosi et al., MSR 2022) converted to the GPT2SP input format.

Two inputs:
  1. The full TAWOS v1.1 MySQL database (loaded locally from the official dump, database `tawos`),
     used only to verify the published size of the dataset -> tawos_db_stats.json and the per-project
     database counts in stats.csv.
  2. Tawosi's cleaned story-point files used in their Deep-SE replication (23 public projects, issues
     sorted by creation time), from github.com/SOLAR-group/AgileEffortEstimation
     (datasets/Tawosi_Dataset/<KEY>_deep-se.csv, commit 291e22a), copied to raw/.

Output: marked_data/<KEY>.csv with the benchmark's columns issuekey,title,description,storypoint,split_mark.
Issues keep Tawosi's chronological order; split_mark is the positional 60/20/20 split GPT2SP applies.
Story points are kept as floats (some projects use 0.5); train_gpt2sp.py trains on integer-cast labels
exactly like the original code but scores against these true values.

Usage: python new_data/tawos/prepare.py [--no-db]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
csv.field_size_limit(sys.maxsize)

REPO = {"MESOS": "Apache", "ALOY": "Apache", "TISTUD": "Appcelerator", "APSTUD": "Appcelerator",
        "CLI": "Appcelerator", "DAEMON": "Appcelerator", "TIDOC": "Appcelerator", "TIMOB": "Appcelerator",
        "CLOV": "Atlassian", "CONFCLOUD": "Atlassian", "CONFSERVER": "Atlassian", "DNN": "DNNSoftware",
        "DURACLOUD": "Duraspace", "FAB": "Hyperledger", "STL": "Hyperledger", "DM": "Lsstcorp",
        "COMPASS": "MongoDB", "SERVER": "MongoDB", "EVG": "MongoDB", "MDL": "Moodle", "MULE": "Mulesoft",
        "NEXUS": "Sonatype", "XD": "Spring"}
# TAWOS key -> GPT2SP benchmark project with the same Jira project (older snapshot, up to Aug 2016)
IN_BENCHMARK = {"MESOS": "mesos", "TISTUD": "appceleratorstudio", "APSTUD": "aptanastudio",
                "TIMOB": "titanium", "CLOV": "clover", "DURACLOUD": "duracloud", "DM": "datamanagement",
                "MDL": "moodle", "MULE": "mule", "XD": "springxd"}


def db_stats():
    import pymysql
    con = pymysql.connect(host="127.0.0.1", user="root", database="tawos")
    q = lambda s: pd.read_sql(s, con)  # noqa: E731
    stats = {
        "issues": int(q("select count(*) n from Issue").n[0]),
        "projects": int(q("select count(*) n from Project").n[0]),
        "repositories": int(q("select count(*) n from Repository").n[0]),
        "issues_with_story_points": int(q("select count(*) n from Issue where Story_Point is not null").n[0]),
    }
    per_proj = q("""select p.Project_Key `key`, count(*) db_issues,
                           sum(i.Story_Point is not null) db_issues_with_sp,
                           min(i.Creation_Date) db_first_created, max(i.Creation_Date) db_last_created
                    from Issue i join Project p on i.Project_ID = p.ID group by p.Project_Key""")
    con.close()
    return stats, per_proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-db", action="store_true", help="skip the MySQL checks")
    args = ap.parse_args()
    out = HERE / "marked_data"
    out.mkdir(exist_ok=True)
    rows = []
    for key in REPO:
        df = pd.read_csv(HERE / "raw" / f"{key}_deep-se.csv", engine="python")
        df["created"] = pd.to_datetime(df["created"], format="%m/%d/%Y %H:%M:%S")
        assert df["created"].is_monotonic_increasing, key
        n = len(df)
        a, b = int(n * 0.6), int(n * 0.8)
        df["split_mark"] = ["train"] * a + ["val"] * (b - a) + ["test"] * (n - b)
        df[["issuekey", "title", "description", "storypoint", "split_mark"]].to_csv(out / f"{key}.csv", index=False)
        sp = df["storypoint"].astype(float)
        rows.append({"key": key, "repository": REPO[key], "in_gpt2sp_benchmark": key in IN_BENCHMARK,
                     "benchmark_name": IN_BENCHMARK.get(key, ""), "issues": n, "n_train": a,
                     "n_val": b - a, "n_test": n - b, "sp_mean": sp.mean(), "sp_median": sp.median(),
                     "sp_min": sp.min(), "sp_max": sp.max(), "frac_non_integer_sp": (sp % 1 != 0).mean(),
                     "first_created": df["created"].min(), "last_created": df["created"].max(),
                     "empty_description": (df["description"].fillna("").str.strip() == "").mean()})
    stats = pd.DataFrame(rows)
    if not args.no_db:
        db, per_proj = db_stats()
        (HERE / "tawos_db_stats.json").write_text(json.dumps(db, indent=2) + "\n")
        stats = stats.merge(per_proj, on="key", how="left")
        print(json.dumps(db))
    stats.to_csv(HERE / "stats.csv", index=False)
    print(stats[["key", "repository", "in_gpt2sp_benchmark", "issues", "n_test", "sp_mean",
                 "frac_non_integer_sp"]].round(2).to_string(index=False))
    print(f"\n{len(stats)} projects, {stats.issues.sum()} issues, "
          f"{(~stats.in_gpt2sp_benchmark).sum()} projects new to GPT2SP")


if __name__ == "__main__":
    main()
