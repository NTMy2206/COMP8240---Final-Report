#!/usr/bin/env bash
# Download the parts of Tawosi et al.'s replication package that this project uses, at the exact
# commit used (SOLAR-group/AgileEffortEstimation @ 291e22a8). The package has no licence file, so
# its code and data are not redistributed in this repository; this script fetches them instead.
#
#   replication/deepse/Deep-SE/                 Deep-SE code, then our 6 changes from upstream.patch
#   replication/deepse/datasets/Choet_Dataset/  the 14-project benchmark in Deep-SE's format
#   replication/deepse/datasets/Pretrain_Dataset/  the 8 repositories those projects need
#   new_data/tawos/raw/                         the 23 cleaned TAWOS projects (*_deep-se.csv)
#
# Usage: bash replication/fetch_tawosi_package.sh      (needs git; ~430 MB download)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT=291e22a8bf812f5a6b3e241e0805159eab48b648
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone -q --filter=blob:none --no-checkout https://github.com/SOLAR-group/AgileEffortEstimation.git "$TMP/aee"
git -C "$TMP/aee" sparse-checkout set --no-cone /Deep-SE/ /datasets/Choet_Dataset/ \
    /datasets/Pretrain_Dataset/ /datasets/Tawosi_Dataset/
git -C "$TMP/aee" checkout -q "$COMMIT"

D="$ROOT/replication/deepse"
if [[ -d "$D/Deep-SE" ]]; then
  echo "replication/deepse/Deep-SE already exists; leaving it as is"
else
  cp -R "$TMP/aee/Deep-SE" "$D/Deep-SE"
  (cd "$D" && patch -p0 --no-backup-if-mismatch < upstream.patch)
fi
mkdir -p "$D/datasets/Pretrain_Dataset"
cp -R "$TMP/aee/datasets/Choet_Dataset" "$D/datasets/"
for f in Apache Appcelerator Atlassian Duraspace Lsstcorp Moodle Mulesoft Spring; do
  cp "$TMP/aee/datasets/Pretrain_Dataset/$f.csv" "$D/datasets/Pretrain_Dataset/"
done
cp "$TMP/aee/datasets/Pretrain_Dataset/README.md" "$D/datasets/Pretrain_Dataset/"

mkdir -p "$ROOT/new_data/tawos/raw"
cp "$TMP/aee/datasets/Tawosi_Dataset/"*_deep-se.csv "$TMP/aee/datasets/Tawosi_Dataset/README.md" "$ROOT/new_data/tawos/raw/"
echo "done: Deep-SE (patched), Choet + Pretrain datasets, 23 TAWOS projects"
