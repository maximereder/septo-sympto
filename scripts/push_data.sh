#!/bin/sh
set -eu

# Push training data into the septosympto-data Modal volume, once.
# The training workers read from the volume and fail fast if it is missing,
# so nothing uploads on a run's hot path.
#
#   scripts/push_data.sh                 # push pycnidia dirs + necrosis zips
#   scripts/push_data.sh pycnidia        # push only the pycnidia dirs
#   scripts/push_data.sh necrosis        # push only the necrosis zips

VOLUME="${SEPTOSYMPTO_DATA_VOLUME:-septosympto-data}"
WHAT="${1:-all}"

put() {
  echo "== $1 -> volume $VOLUME at /$2 =="
  uvx modal volume put --force "$VOLUME" "$1" "/$2"
}

if [ "$WHAT" = "pycnidia" ] || [ "$WHAT" = "all" ]; then
  for d in data/pycnidia/train-50-aug-x3 data/pycnidia/train-100-aug-x3 \
           data/pycnidia/train-200-aug-x3 data/pycnidia/valid-40; do
    [ -d "$d" ] && put "$d" "${d#data/}"
  done
fi

if [ "$WHAT" = "necrosis" ] || [ "$WHAT" = "all" ]; then
  for z in data/necrosis/dataset/*.zip; do
    [ -f "$z" ] && put "$z" "${z#data/}"
  done
fi

echo "done — data is in volume '$VOLUME'"
