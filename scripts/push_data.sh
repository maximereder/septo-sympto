#!/bin/sh
set -eu

# Push training data into the septosympto-data Modal volume, once.
# The training workers read from the volume and fail fast if it is missing,
# so nothing uploads on a run's hot path.
#
#   scripts/push_data.sh                 # push everything below
#   scripts/push_data.sh native          # push only the native letterbox set
#   scripts/push_data.sh pycnidia        # push only the Roboflow pycnidia dirs
#   scripts/push_data.sh necrosis        # push only the necrosis zips

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

VOLUME="${SEPTOSYMPTO_DATA_VOLUME:-septosympto-data}"
WHAT="${1:-all}"

put() {
  echo "== $1 -> volume $VOLUME at /$2 =="
  uvx modal volume put --force "$VOLUME" "$1" "/$2"
}

if [ "$WHAT" = "native" ] || [ "$WHAT" = "all" ]; then
  [ -d data/leaves-native ] && put data/leaves-native leaves-native
fi

if [ "$WHAT" = "necrosis" ] || [ "$WHAT" = "all" ]; then
  for z in data/necrosis/dataset/*.zip; do
    [ -f "$z" ] && put "$z" "${z#data/}"
  done
fi

echo "done — data is in volume '$VOLUME'"
