#!/usr/bin/env bash
# Run GPT2SP and the naive baselines on the two new datasets.
#   1. TAWOS (existing dataset, Tawosi's cleaned 23-project version)
#   2. Recent Moodle issues (newly constructed, mined from Moodle's public Jira)
# Usage: bash replication/run_new_data.sh [--collect] [--seeds "1 2 3"]
#   --collect   re-download the Moodle issues first (otherwise uses new_data/moodle_recent/raw)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SEEDS="1 2 3"
COLLECT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --collect) COLLECT=1; shift ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
TAWOS="MESOS ALOY TISTUD APSTUD CLI DAEMON TIDOC TIMOB CLOV CONFCLOUD CONFSERVER DNN DURACLOUD FAB STL DM COMPASS SERVER EVG MDL MULE NEXUS XD"

echo "=== TAWOS ==="
ls new_data/tawos/raw/*_deep-se.csv >/dev/null 2>&1 || bash replication/fetch_tawosi_package.sh
$PY new_data/tawos/prepare.py || $PY new_data/tawos/prepare.py --no-db
$PY replication/naive_baselines.py --scenario within --data-dir new_data/tawos/marked_data \
    --projects $TAWOS --tag tawos_within
$PY replication/train_gpt2sp.py --scenario within --data-dir new_data/tawos/marked_data \
    --projects $TAWOS --seeds $SEEDS --tag tawos_within --skip-done

echo "=== Recent Moodle ==="
if [[ $COLLECT == 1 ]]; then $PY new_data/moodle_recent/collect.py; fi
$PY new_data/moodle_recent/preprocess.py
$PY new_data/annotation/sample_for_annotation.py
# (a) train and test on recent issues (60/20/20 by time)
$PY replication/naive_baselines.py --scenario within --projects moodle_recent \
    --data-dir new_data/moodle_recent/marked_data --tag moodle_recent_within
$PY replication/train_gpt2sp.py --scenario within --projects moodle_recent \
    --data-dir new_data/moodle_recent/marked_data --seeds $SEEDS --tag moodle_recent_within --skip-done
# (b) train on the 2016 benchmark Moodle project (60/40 train/val), test on every recent issue
$PY replication/naive_baselines.py --scenario custom --train moodle --test moodle_recent \
    --data-dir new_data/moodle_recent/marked_data --tag moodle_2016_to_recent
$PY replication/train_gpt2sp.py --scenario custom --train moodle --test moodle_recent \
    --data-dir new_data/moodle_recent/marked_data --seeds $SEEDS --tag moodle_2016_to_recent --skip-done
echo "done"
