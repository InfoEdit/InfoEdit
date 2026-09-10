#!/usr/bin/env bash
# Run Qwen-Image-Edit-2511 locally on one InfoEdit infographic.
#
# Point HF_HOME at wherever you want model weights cached:
#   export HF_HOME=/path/to/hf-cache
#
# Batch usage: loop over the ids you want, feeding the rendered PNG from
# data/html_infographics/v17/ and the matching instruction from
# data/editing_prompts_html/v17.<task>.jsonl, and write results to
# edited_html_infographics/v17/<model>/<task>/{id}_edited_1.png so that
# run_eval.sh can score them.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-../../../data/html_infographics/v17/1.png}"
OUTPUT="${OUTPUT:-1_edited_1.png}"
PROMPT="${PROMPT:?set PROMPT to the editing instruction}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python edit.py \
    --image "$IMAGE" \
    --prompt "$PROMPT" \
    --output "$OUTPUT"
