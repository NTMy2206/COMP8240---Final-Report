"""Script port of model_training_notebook.ipynb (GPT2SP, BPE GPT-2 tokenizer, model #0/#00/#000).

Same model class (GPT2SP.py), hyperparameters, title-only input,
positional 60/20/20 within-project split, 60/40 train/val split of the source project for
cross-project, dynamic batch size, L1 loss, AdamW + linear schedule, and model selection by
lowest validation loss (ties -> later epoch).

Mechanical changes only: device is cuda -> mps -> cpu instead of hard-coded .cuda();
large batches are computed in micro-batches with exact gradient accumulation (same gradient,
one optimizer step per notebook batch) so they fit in Apple-GPU memory;
pd.concat instead of DataFrame.append; torch.optim.AdamW instead of the removed
transformers.AdamW; seeds are fixed; per-issue predictions of the selected epoch are saved so
MAE can be recomputed from raw predictions (replication/metrics.py).

Labels: as in the notebook, training/validation labels are cast to LongTensor (so a 0.5-point
label is trained as 0). Test predictions are scored against the TRUE (float) labels, which only
matters for datasets with fractional story points (TAWOS).

Usage:
  python replication/train_gpt2sp.py --scenario within --seeds 1 2 3
  python replication/train_gpt2sp.py --scenario custom --train moodle --test moodle_recent \
      --data-dir new_data/moodle_recent/marked_data
"""
import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import GPT2Config, GPT2Tokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "replication"))
from GPT2SP import GPT2ForSequenceClassification as GPT2SP  # noqa: E402
import replication.metrics as metrics  # noqa: E402

# ---- hyperparameters (unchanged from the notebook) ----
EPOCHS = 20
SEQUENCE_LEN = 20
LEARNING_RATE = 5e-4
BATCH_RATIO_WITHIN = 0.3   # within proj: 0.3 / cross proj: 0.4
BATCH_RATIO_CROSS = 0.4
PAD_TOKEN_ID = 50256
DEFAULT_DATA_DIR = ROOT / "sp_dataset" / "marked_data"

BENCHMARK = ["appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement",
             "duracloud", "jirasoftware", "mesos", "moodle", "mule", "mulestudio", "springxd",
             "talenddataquality", "talendesb", "titanium", "usergrid"]
WITHIN_REPO_PAIRS = [("mesos", "usergrid"), ("usergrid", "mesos"),
                     ("appceleratorstudio", "aptanastudio"), ("appceleratorstudio", "titanium"),
                     ("titanium", "appceleratorstudio"), ("aptanastudio", "titanium"),
                     ("mule", "mulestudio"), ("mulestudio", "mule")]
CROSS_REPO_PAIRS = [("clover", "usergrid"), ("talendesb", "mesos"),
                    ("talenddataquality", "aptanastudio"), ("mule", "titanium"),
                    ("talenddataquality", "appceleratorstudio"), ("mulestudio", "titanium"),
                    ("appceleratorstudio", "mulestudio"), ("appceleratorstudio", "mule")]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_TOKENIZER = None


def tokenization(text_list):
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
        _TOKENIZER.pad_token = "[PAD]"   # not in vocab -> maps to id 50256, as in the notebook
    return _TOKENIZER.batch_encode_plus(text_list, truncation=True, max_length=SEQUENCE_LEN,
                                        padding="max_length")


def find_csv(name, data_dirs):
    for d in data_dirs:
        p = Path(d) / f"{name}.csv"
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}.csv not found in {[str(d) for d in data_dirs]}")


def prepare_dataframe(path):
    data = pd.read_csv(path)
    data = data.fillna(" ")  # some rows have no description, fill blank to avoid Null
    return pd.DataFrame({"issuekey": data["issuekey"].tolist(),
                         "text": data["title"].astype(str).tolist(),
                         "label": data["storypoint"].astype(float).tolist()})


def to_tensors(df):
    seq = torch.tensor(tokenization(df["text"].tolist())["input_ids"])
    y = torch.tensor(df["label"].tolist()).type(torch.LongTensor)  # notebook: integer labels
    return seq, y


def make_loader(seq, y, batch_size, random_order):
    ds = TensorDataset(seq, y)
    sampler = RandomSampler(ds) if random_order else SequentialSampler(ds)
    return DataLoader(ds, sampler=sampler, batch_size=batch_size)


def build_data(train_names, test_name, within, data_dirs):
    frames = [prepare_dataframe(find_csv(n, data_dirs)) for n in train_names]
    data = pd.concat(frames, ignore_index=True)
    if within:
        a, b = int(len(data) * 0.6), int(len(data) * 0.8)
        train, val, test = data.iloc[:a], data.iloc[a:b], data.iloc[b:]
        ratio = BATCH_RATIO_WITHIN
    else:
        a = int(len(data) * 0.6)
        train, val = data.iloc[:a], data.iloc[a:]
        test = prepare_dataframe(find_csv(test_name, data_dirs))
        ratio = BATCH_RATIO_CROSS
    batch_size = int(len(train) * ratio)
    return train, val, test.reset_index(drop=True), batch_size


def forward_loss(model, ids, labels, device, micro):
    """Mean L1 loss over a notebook batch, computed in micro-batches.

    Returns the batch-mean loss (float). If grad is enabled, gradients equal those of a single
    forward/backward pass over the whole batch (each micro loss is weighted by its share)."""
    n = ids.shape[0]
    total = 0.0
    for s in range(0, n, micro):
        mi, ml = ids[s:s + micro].to(device), labels[s:s + micro].to(device)
        out = model(mi, labels=ml, return_dict=True)
        w = mi.shape[0] / n
        if torch.is_grad_enabled():
            (out.loss * w).backward()
        total += out.loss.item() * w
    return total


@torch.no_grad()
def predict(model, ids, device, micro):
    preds = []
    for s in range(0, ids.shape[0], micro):
        preds.append(model(ids[s:s + micro].to(device))["logits"].float().cpu().numpy().reshape(-1))
    return np.concatenate(preds)


def run_one(train_names, test_name, within, seed, data_dirs, out_dir, device, micro,
            tag, save_model=None, log=print):
    set_seed(seed)
    config = GPT2Config(num_labels=1, pad_token_id=PAD_TOKEN_ID)
    model = GPT2SP.from_pretrained("gpt2", config=config).to(device)

    train, val, test, batch_size = build_data(train_names, test_name, within, data_dirs)
    tr_seq, tr_y = to_tensors(train)
    va_seq, va_y = to_tensors(val)
    te_seq, _ = to_tensors(test)
    te_true = test["label"].to_numpy(dtype=float)
    train_loader = make_loader(tr_seq, tr_y, batch_size, random_order=True)
    val_loader = make_loader(va_seq, va_y, batch_size, random_order=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=total_steps)
    best = [10000.0, 0]
    epoch_rows, epoch_preds, time_records = [], [], []
    start = time.time()
    for e in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        for ids, labels in train_loader:
            model.zero_grad()
            tr_loss += forward_loss(model, ids, labels, device, micro)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        tr_loss /= len(train_loader)
        time_records.append(time.time() - start)

        model.eval()
        with torch.no_grad():
            va_loss = sum(forward_loss(model, ids, labels, device, micro)
                          for ids, labels in val_loader) / len(val_loader)
        if va_loss <= best[0]:
            best = [va_loss, e]
        preds = predict(model, te_seq, device, micro)
        epoch_preds.append(preds)
        te_mae = metrics.mae(te_true, preds)
        epoch_rows.append({"epoch": e, "train_loss": tr_loss, "val_loss": va_loss, "test_mae": te_mae})
        log(f"  epoch {e:2d}  train MAE loss {tr_loss:6.2f}  val MAE loss {va_loss:6.2f}  test MAE {te_mae:6.2f}")
        if save_model is not None and best[1] == e:
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    be = best[1]
    preds = epoch_preds[be]
    s = metrics.summarise(te_true, preds, batch_size)
    row = {"tag": tag, "train": "+".join(train_names), "test": test_name, "seed": seed,
           "n_train": len(train), "n_val": len(val), "n_test": len(test), "batch_size": batch_size,
           "best_epoch": be, "best_val_loss": best[0], "mae": s["mae"], "mdae": s["mdae"],
           "mae_inspection_formula": s["mae_inspection_formula"],
           "train_time_s": time_records[be], "total_time_s": time.time() - start,
           "device": str(device)}
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{'+'.join(train_names)}__{test_name}__seed{seed}"
    pd.DataFrame({"issuekey": test["issuekey"], "actual": te_true, "predicted": preds}).to_csv(
        out_dir / f"{stem}_preds.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(out_dir / f"{stem}_epochs.csv", index=False)
    runs = out_dir / "runs.csv"
    pd.DataFrame([row]).to_csv(runs, mode="a", header=not runs.exists(), index=False)
    log(f"<<< MAE {s['mae']:.2f}  MdAE {s['mdae']:.2f}  (inspection-notebook formula: "
        f"{s['mae_inspection_formula']:.2f})  best epoch {be}  {time_records[be]:.1f}s")
    if save_model is not None:
        model.load_state_dict(best_state)
        model.save_pretrained(Path(save_model) / stem)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=["within", "within_repo", "cross_repo", "custom"], required=True)
    ap.add_argument("--projects", nargs="*", help="within: projects (default: the 16 benchmark projects)")
    ap.add_argument("--train", nargs="*", help="custom: training project(s)")
    ap.add_argument("--test", help="custom: test project")
    ap.add_argument("--custom-within", action="store_true",
                    help="custom: use the within-project 60/20/20 split of --train instead of 60/40 + external test")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--data-dir", action="append", default=[],
                    help="extra directory with <project>.csv files (searched before sp_dataset/marked_data)")
    ap.add_argument("--out", default=None, help="output dir (default results/gpt2sp/<scenario or tag>)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--micro-batch", type=int, default=128)
    ap.add_argument("--save-model", default=None, help="directory to save the selected checkpoint")
    ap.add_argument("--skip-done", action="store_true", help="skip runs already in runs.csv")
    args = ap.parse_args()

    data_dirs = [ROOT / d if not os.path.isabs(d) else Path(d) for d in args.data_dir] + [DEFAULT_DATA_DIR]
    device = pick_device()
    if args.scenario == "within":
        jobs = [([p], p, True) for p in (args.projects or BENCHMARK)]
    elif args.scenario == "within_repo":
        jobs = [([a], b, False) for a, b in WITHIN_REPO_PAIRS]
    elif args.scenario == "cross_repo":
        jobs = [([a], b, False) for a, b in CROSS_REPO_PAIRS]
    else:
        if not args.train:
            ap.error("--scenario custom needs --train (and --test unless --custom-within)")
        jobs = [(args.train, args.test or args.train[0], args.custom_within)]
    tag = args.tag or args.scenario
    out_dir = Path(args.out) if args.out else ROOT / "results" / "gpt2sp" / tag
    done = set()
    if args.skip_done and (out_dir / "runs.csv").exists():
        r = pd.read_csv(out_dir / "runs.csv")
        done = set(zip(r["train"], r["test"], r["seed"]))
    print(f"device: {device.type}  scenario: {args.scenario}  pairs: {len(jobs)}  seeds: {args.seeds}")
    for seed in args.seeds:
        for train_names, test_name, within in jobs:
            if ("+".join(train_names), test_name, seed) in done:
                continue
            print(f"\n>>> train {'+'.join(train_names)} -> test {test_name}  (seed {seed})", flush=True)
            run_one(train_names, test_name, within, seed, data_dirs, out_dir, device,
                    args.micro_batch, tag, args.save_model, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
