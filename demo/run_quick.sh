#!/usr/bin/env bash
# Live demo: retrain GPT2SP on one small project (Clover, ~30 s on an Apple-silicon GPU) and show
# the correct MAE next to the paper's inspection-notebook formula on the same predictions.
# Usage: bash demo/run_quick.sh [project] [seed]
set -euo pipefail
cd "$(dirname "$0")/.."
P="${1:-clover}"; SEED="${2:-1}"
rm -f demo/output/runs.csv
.venv/bin/python replication/train_gpt2sp.py --scenario within --projects "$P" --seeds "$SEED" \
    --out demo/output --tag demo 2>&1 | grep -vE "attention_mask|pad_token_id|newly initialized|TRAIN this model|^$"
echo
echo "First test predictions (demo/output/${P}__${P}__seed${SEED}_preds.csv):"
head -6 "demo/output/${P}__${P}__seed${SEED}_preds.csv" | column -t -s,
