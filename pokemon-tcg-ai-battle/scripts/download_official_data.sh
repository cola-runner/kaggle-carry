#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

KAGGLE_BIN="${KAGGLE_BIN:-}"
if [[ -z "$KAGGLE_BIN" ]]; then
  if [[ -x ".venv/bin/kaggle" ]]; then
    KAGGLE_BIN=".venv/bin/kaggle"
  else
    KAGGLE_BIN="kaggle"
  fi
fi

mkdir -p data/raw

"$KAGGLE_BIN" competitions files -c pokemon-tcg-ai-battle
"$KAGGLE_BIN" competitions download -c pokemon-tcg-ai-battle -p data/raw

if [[ -f data/raw/pokemon-tcg-ai-battle.zip ]]; then
  unzip -o data/raw/pokemon-tcg-ai-battle.zip -d data/raw/pokemon-tcg-ai-battle
fi
