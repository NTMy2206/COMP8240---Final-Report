"""Live demo: the authors' released Mule model on the paper's example issue MULE-14597.

Shows the BPE word pieces GPT2SP sees (title only, at most 20) and the story-point estimate.
Usage: .venv/bin/python demo/predict_one.py ["any issue title"] [--model MickyMike/0-GPT2SP-mule]
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import GPT2Tokenizer
from transformers.utils import logging

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from GPT2SP import GPT2ForSequenceClassification as GPT2SP  # noqa: E402

logging.set_verbosity_error()
ap = argparse.ArgumentParser()
ap.add_argument("title", nargs="?", default=None)
ap.add_argument("--model", default="MickyMike/0-GPT2SP-mule")
args = ap.parse_args()

# The paper's running example (Fu & Tantithamthavorn 2023, Sec. 2): not part of the 2016 benchmark CSVs.
EXAMPLE_TITLE = "ExpressionExecutionException when element that expects a DW starts with space"
EXAMPLE_SP = 4
title = args.title or EXAMPLE_TITLE
tok = GPT2Tokenizer.from_pretrained("gpt2")
tok.pad_token = "[PAD]"
enc = tok.batch_encode_plus([title], truncation=True, max_length=20, padding="max_length")
pieces = [t.replace("Ġ", "") for t in tok.convert_ids_to_tokens(enc["input_ids"][0])
          if t != tok.eos_token][: sum(i != 50256 for i in enc["input_ids"][0])]
model = GPT2SP.from_pretrained(args.model).eval()
with torch.no_grad():
    est = model(torch.tensor(enc["input_ids"]))["logits"].item()
print(f"model      : {args.model}")
print(f"issue title: {title}")
print(f"BPE tokens : {'|'.join(pieces)}  ({len(pieces)} of max 20)")
print(f"estimate   : {est:.2f} story points")
if args.title is None:
    print(f"recorded   : {EXAMPLE_SP} story points (MULE-14597, the paper's example)")
