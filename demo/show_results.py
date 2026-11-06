"""Print the headline tables (built by replication/build_tables.py and replication/scottknott.R)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "results" / "tables"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


def show(title, path, cols=None):
    if not path.exists():
        print(f"\n=== {title} ===\n(missing: {path.relative_to(ROOT)})")
        return
    df = pd.read_csv(path)
    print(f"\n=== {title} ===")
    print((df[cols] if cols else df).round(2).to_string(index=False))


show("Replication, within-project MAE (16 projects)", T / "within_project.csv",
     ["project", "paper_gpt2sp", "gpt2sp_buggy_formula", "tawosi_gpt2sp", "our_gpt2sp", "gpt2sp_sd",
      "tawosi_deepse", "our_deepse", "median_mae", "mean_mae"])
show("Authors' released checkpoints, re-scored",
     ROOT / "results" / "published_checkpoints" / "published_checkpoints.csv",
     ["project", "n_test", "batch_size", "mae", "inspection_notebook_mae"])
sk = T / "scottknott_within.csv"
if sk.exists():
    d = pd.read_csv(sk)
    print("\n=== Scott-Knott ESD (rank 1 = best group) ===")
    print(d[d.project.str.startswith("ALL")].round(2).to_string(index=False))
show("Cross-project MAE", T / "cross_project.csv")
show("New data 1: Tawosi/TAWOS dataset (23 projects)", T / "tawos_within.csv")
show("New data 2: recent Moodle issues (collected 2026)", T / "moodle_recent.csv")
show("Moodle collection funnel", ROOT / "new_data" / "moodle_recent" / "funnel.csv")
st = json.loads((ROOT / "new_data" / "moodle_recent" / "stats.json").read_text())
print("\nStory points, benchmark 2016 vs recent:", st["storypoints_benchmark_2016"], "vs",
      st["storypoints_recent"])
