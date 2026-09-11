#!/usr/bin/env bash
# InfoEdit quick start — end-to-end on a handful of examples (~2 min of compute).
#
#   cp env.example.sh env.sh && $EDITOR env.sh && source env.sh
#   bash quickstart.sh
#
# Runs one editor over 5 Expand-Text examples, then scores them.
set -euo pipefail
cd "$(dirname "$0")"

export MODEL="${MODEL:-gemini-3.5-flash}"   # a text model: this branch rewrites source
export TASK="${TASK:-text_expand}"
export SOURCE="${SOURCE:-html}"
export LIMIT="${LIMIT:-5}"

case "$SOURCE" in
  html) VERSION=v17 ;;
  ppt)  VERSION=v7  ;;
esac
if [ ! -f "data/editing_prompts_${SOURCE}/${VERSION}.${TASK}.jsonl" ]; then
  echo "Benchmark data not found: data/editing_prompts_${SOURCE}/${VERSION}.${TASK}.jsonl"
  echo "Fetch the benchmark first:"
  echo "  huggingface-cli download InfoEdit/InfoEdit --repo-type dataset --local-dir data"
  exit 1
fi

echo "==> 1/2  editing ${LIMIT} examples with ${MODEL}"
bash run_edit.sh

echo "==> 2/2  scoring with the reflow-aware judge"
bash run_eval.sh

echo ""
echo "Done. Detailed per-example judgements: eval_results/${MODEL}/"
