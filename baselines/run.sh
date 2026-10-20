#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

CHMCNN_COMMIT=e887855ab4b3
SPUR_COMMIT=bd6055c9e01d
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-3}
OUT=${OUT:-out}

[ -d C-HMCNN ] || { git clone -q https://github.com/EGiunchiglia/C-HMCNN.git && git -C C-HMCNN checkout -q "$CHMCNN_COMMIT"; }
[ -d SPUR ] || { git clone -q https://github.com/yuanzx33033/SPUR.git && git -C SPUR checkout -q "$SPUR_COMMIT"; }
mkdir -p "$OUT/logs" "$OUT/done" "$OUT/chmcnn" "$OUT/spur"
export OUT DATA_FOLDER="$PWD/C-HMCNN"

run_job() {
  local tag="$1-$2-$3"
  [ -f "$OUT/done/$tag" ] && return 0
  python authors_code.py "$1" "$2" "$3" > "$OUT/logs/$tag.log" 2>&1 && touch "$OUT/done/$tag"
}
export -f run_job

for seed in $SEEDS; do
  for d in cellcycle derisi eisen expr gasch1 gasch2 seq spo; do
    echo "C-HMCNN ${d}_FUN $seed"
    echo "C-HMCNN ${d}_GO $seed"
    echo "SPUR ${d}_FUN $seed"
  done
  for d in diatoms enron imclef07a imclef07d; do
    echo "C-HMCNN ${d}_others $seed"
  done
done | xargs -P "$JOBS" -L 1 bash -c 'run_job "$@"' _

cp C-HMCNN/results/*.csv "$OUT/chmcnn/" 2>/dev/null
cp SPUR/results/*.csv "$OUT/spur/" 2>/dev/null
echo "$(ls "$OUT/done" | wc -l) runs finished"
