"""Re-score the authors' released within-project checkpoints (HF Hub MickyMike/0-GPT2SP-<project>).

For each project the test set is the newest 20% of sp_dataset/marked_data/<project>.csv, exactly as
in GPT2SP_inspection_notebook.ipynb. Predictions are saved per issue and scored twice:

  mae                      correct MAE over all test issues
  inspection_notebook_mae  the notebook formula, sum|e| / (n_batches * len(first batch)),
                           with the notebook's batch size int(int(0.6*n) * 0.3)

The second number reproduces the values published in the paper.

Only change from the notebook: its data path sp_dataset/marked_training_data/ does not exist in
the repository, so sp_dataset/marked_data/ is used. Inference runs on CPU by default.

Usage: python replication/check_published_checkpoints.py [--device cpu|mps]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "replication"))
from GPT2SP import GPT2ForSequenceClassification as GPT2SP  # noqa: E402
import replication.metrics as metrics  # noqa: E402
from replication.train_gpt2sp import BENCHMARK, DEFAULT_DATA_DIR, prepare_dataframe, tokenization  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "results" / "published_checkpoints"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    rows = []
    for p in BENCHMARK:
        df = prepare_dataframe(DEFAULT_DATA_DIR / f"{p}.csv")
        batch_size = int(int(len(df["text"][: int(len(df) * 0.6)])) * 0.3)
        test = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)
        ids = torch.tensor(tokenization(test["text"].tolist())["input_ids"])
        model = GPT2SP.from_pretrained(f"MickyMike/0-GPT2SP-{p}").to(args.device).eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(ids), batch_size):
                preds.append(model(ids[s:s + batch_size].to(args.device))["logits"].cpu().reshape(-1))
        pred = torch.cat(preds).numpy()
        actual = test["label"].to_numpy(dtype=float)
        pd.DataFrame({"issuekey": test["issuekey"], "actual": actual, "predicted": pred}).to_csv(
            out / "preds" / f"{p}_preds.csv", index=False)
        row = {"project": p, "n_test": len(test), "batch_size": batch_size,
               "mae": metrics.mae(actual, pred), "mdae": metrics.mdae(actual, pred),
               "inspection_notebook_mae": metrics.inspection_notebook_mae(actual, pred, batch_size)}
        rows.append(row)
        print(f"{p:20s} n_test {len(test):4d}  batch {batch_size:4d}  MAE {row['mae']:.2f}  "
              f"inspection formula {row['inspection_notebook_mae']:.2f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "published_checkpoints.csv", index=False)
    print(f"\nmedian MAE {res['mae'].median():.2f}   median inspection-formula MAE "
          f"{res['inspection_notebook_mae'].median():.2f}")


if __name__ == "__main__":
    main()
