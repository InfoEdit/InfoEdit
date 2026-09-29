#!/usr/bin/env bash
# Step 1 — run an editor over InfoEdit and write edited infographics.
#
#   MODEL=gemini-2.5-flash-image TASK=add bash scripts/run_edit.sh
#
# Environment variables (all optional except MODEL):
#   MODEL    editor model id                                  (required)
#   TASK     text_expand | add | swap_inter | aspect_ratio     (default text_expand)
#   SOURCE   html | ppt                                        (default html)
#   PATHWAY  image | code | code_image | gpt | seedream        (default image)
#   BACKEND  vertex | openai (proxy; code pathways only)       (default vertex)
#   LIMIT    number of examples; empty = all                   (default all)
cd "$(dirname "$0")/.."          # repo root: data/ and outputs are relative to it
source scripts/_common.sh

python "$EDITOR" \
    --input_file "$PROMPT_FILE" \
    --output_dir "$EDITED_DIR" \
    --model_path "$MODEL" \
    --operation "$TASK" \
    --num_workers "$WORKERS" \
    "${BACKEND_ARGS[@]}" \
    --no_wandb \
    ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} ${EXTRA_EDIT_ARGS[@]+"${EXTRA_EDIT_ARGS[@]}"}

echo "Edits written to ${EDITED_DIR}/ — now run: MODEL=$MODEL TASK=$TASK bash scripts/run_eval.sh"
