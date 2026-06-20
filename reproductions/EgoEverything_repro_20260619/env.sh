#!/usr/bin/env bash

export EGOEVERYTHING_BASE="${EGOEVERYTHING_BASE:-/scratch/xy3257/EgoEverything_repro_20260619}"
export EGOEVERYTHING_REPO="${EGOEVERYTHING_REPO:-$EGOEVERYTHING_BASE/EgoEverything}"

if [[ ! -d "$EGOEVERYTHING_BASE/.venv" ]]; then
  echo "Missing virtualenv: $EGOEVERYTHING_BASE/.venv" >&2
  return 1 2>/dev/null || exit 1
fi

source "$EGOEVERYTHING_BASE/.venv/bin/activate"

export HOME="$EGOEVERYTHING_BASE/home"
export MPLCONFIGDIR="$EGOEVERYTHING_BASE/mplconfig"
export XDG_CACHE_HOME="$EGOEVERYTHING_BASE/cache"
export TORCH_HOME="$EGOEVERYTHING_BASE/cache/torch"
export PYTHONPYCACHEPREFIX="$EGOEVERYTHING_BASE/pycache"
export PATH="$EGOEVERYTHING_BASE/.venv/bin:$PATH"

export DATASET_PATH="${DATASET_PATH:-$EGOEVERYTHING_BASE/Data}"
export DATASET_JSON="${DATASET_JSON:-$DATASET_PATH/AriaEverydayActivities_download_urls.json}"
export DATASET_NAME="${DATASET_NAME:-AriaEveryday_Activities}"

mkdir -p "$HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PYTHONPYCACHEPREFIX" "$DATASET_PATH" "$EGOEVERYTHING_BASE/logs"

cd "$EGOEVERYTHING_REPO" || return 1 2>/dev/null || exit 1
