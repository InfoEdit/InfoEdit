#!/usr/bin/env bash
# InfoEdit quick start — end-to-end on a handful of examples (~2 min of compute).
#
#   cp scripts/env.example.sh env.sh && $EDITOR env.sh && source env.sh
#   bash scripts/quickstart.sh
#
# Runs one editor over 5 Expand-Text examples, then scores them.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

if [ "${BACKEND:-vertex}" = openai ]; then
  # The proxy backend only rewrites source, so default to a text model.
  export PATHWAY="${PATHWAY:-code}"
  export MODEL="${MODEL:-gpt-5.6-sol}"
fi
export MODEL="${MODEL:-gemini-2.5-flash-image}"
export TASK="${TASK:-text_expand}"
export SOURCE="${SOURCE:-html}"
export LIMIT="${LIMIT:-5}"

if [ ! -f "data/editing_prompts/${SOURCE}.${TASK}.jsonl" ]; then
  echo "Benchmark data not found: data/editing_prompts/${SOURCE}.${TASK}.jsonl"
  echo "Fetch the benchmark first:"
  echo "  huggingface-cli download InfoEdit/InfoEdit --repo-type dataset --local-dir data"
  exit 1
fi

echo "==> 1/2  editing ${LIMIT} examples with ${MODEL}"
bash scripts/run_edit.sh

echo "==> 2/2  scoring with the reflow-aware judge"
bash scripts/run_eval.sh

echo ""
case "${PATHWAY:-image}" in
  code|code_image) TAG="${MODEL}_${PATHWAY}" ;;   # same run tag as _common.sh
  *)               TAG="${MODEL}" ;;
esac
echo "Done. Detailed per-example judgements: eval_results/${TAG}/"
