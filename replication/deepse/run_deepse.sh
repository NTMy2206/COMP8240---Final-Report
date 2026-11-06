#!/usr/bin/env bash
# Re-run Tawosi et al.'s Deep-SE replication package (upstream commit 291e22a8, SOLAR-group/AgileEffortEstimation)
# on the 14-project Choet benchmark, "Deep-SE with pre-training" configuration
# (lstm + highway, dim 50, inphid dropout, finetune_lm, mean pooling, maxlen 100, seed 1234).
#
# Usage:  ./run_deepse.sh [stage ...]       stages: fetch build pretrain choet lstm2vec train collect   (default: all)
#   fetch downloads the upstream code + data and applies upstream.patch (skipped if already present)
#   PARALLEL=N   number of projects trained concurrently in the train stage (default 7; each uses 1 CPU thread)
#   PROJECTS="CLOV ..." subset of Tawosi keys for the train stage (default: all 14)
#   EXP=name     experiment directory name under Deep-SE/classification (default choet_pre)
# Everything is written inside this directory, plus results to ../../results/deepse.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RESULTS="$ROOT/results/deepse"
LOGS="$RESULTS/logs"
IMAGE=deepse:py27
PARALLEL="${PARALLEL:-7}"
EXP="${EXP:-choet_pre}"
PROJECTS="${PROJECTS:-CLOV USERGRID MESOS TISTUD APSTUD TIMOB BAM JRESERVER DURACLOUD DM MDL MULE MULESTUDIO XD}"
mkdir -p "$LOGS"

drun() {  # drun <workdir relative to Deep-SE> <command...>
  local wd="$1"; shift
  docker run --rm --platform linux/amd64 -v "$HERE":/work -w "/work/Deep-SE/$wd" "$IMAGE" "$@"
}

stage_fetch() {  # upstream code and data are not redistributed (no licence); download them
  if [[ -d "$HERE/Deep-SE" && -d "$HERE/datasets/Choet_Dataset" ]]; then echo "already fetched"; return; fi
  bash "$ROOT/replication/fetch_tawosi_package.sh"
}

stage_build() {
  docker build --platform linux/amd64 -t "$IMAGE" "$HERE"
}

stage_pretrain() {  # tokenise pre-training repos + build per-repo dictionaries (data/files/<Repo>.dict.pkl.gz)
  drun data python run_script.py Pretrain_Dataset 2>&1 | tee "$LOGS/1_data_pretrain.log"
}

stage_choet() {  # 60/20/20 split by creation order + tokenise with the repo dictionaries
  drun data python run_script.py Choet_Dataset 2>&1 | tee "$LOGS/2_data_choet.log"
}

stage_lstm2vec() {  # NCE lstm2vec features from the shipped language models (NCE/bestModels)
  drun NCE python exp_lstm2v.py lstm2vec Choet_Dataset 2>&1 | tee "$LOGS/3_nce_lstm2vec.log"
}

stage_train() {  # classification/exp_script.py, one container per project, PARALLEL at a time
  rm -rf "$HERE/Deep-SE/classification/$EXP"   # baseline random-guess cache would otherwise change the RNG stream
  for p in $PROJECTS; do
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
    (
      start=$(date +%s); rc=0
      drun classification python exp_script.py "$EXP" "$p" > "$LOGS/4_train_${EXP}_$p.log" 2>&1 || rc=$?
      echo "$EXP $p exit=$rc seconds=$(( $(date +%s) - start ))" >> "$LOGS/4_train_status.txt"
    ) &
  done
  wait
}

stage_collect() {  # per-issue predictions + MAE table
  python3 "$HERE/collect_results.py" --deepse "$HERE/Deep-SE" --exp "$EXP" --data "$HERE/datasets/Choet_Dataset" --out "$RESULTS"
}

STAGES="${*:-fetch build pretrain choet lstm2vec train collect}"
for s in $STAGES; do
  echo "=== stage $s  $(date '+%F %T')"
  SECONDS=0
  "stage_$s"
  echo "=== stage $s done in ${SECONDS}s"
done
