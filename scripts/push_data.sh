#!/bin/sh
set -eu

# Push training data into the septosympto-data Modal volume, once.
# The training workers read from the volume and fail fast if it is missing,
# so nothing uploads on a run's hot path.
#
# Runs modal through Poetry, like every other command in the repository, so it
# needs the train group: poetry install --with train
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
  poetry run modal volume put --force "$VOLUME" "$1" "/$2"
}

# `put --force` overwrites but never deletes, so a regenerated annotation set
# would be unioned with the stale one on the volume. Prune first; img/ is left
# in place (large, and overwritten file by file anyway).
prune() {
  echo "== prune volume $VOLUME /$1 =="
  poetry run modal volume rm -r "$VOLUME" "/$1" 2>/dev/null || true
}

if [ "$WHAT" = "native" ] || [ "$WHAT" = "all" ]; then
  if [ -d data/leaves-native ]; then
    prune leaves-native/mask
    prune leaves-native/labels
    put data/leaves-native leaves-native
  fi
fi

if [ "$WHAT" = "necrosis" ] || [ "$WHAT" = "all" ]; then
  for z in data/necrosis/dataset/*.zip; do
    [ -f "$z" ] && put "$z" "${z#data/}"
  done
fi

echo "done — data is in volume '$VOLUME'"
