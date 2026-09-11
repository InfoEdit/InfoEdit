#!/usr/bin/env bash
# Step 1 — run an editor over InfoEdit and write edited infographics.
#
#   MODEL=gemini-2.5-flash-image TASK=add bash run_edit.sh
#
# Environment variables (all optional except MODEL):
#   MODEL    editor model id                                  (required)
#   TASK     text_expand | add | swap_inter | aspect_ratio     (default text_expand)
#   SOURCE   html | ppt                                        (default html)
#   PATHWAY  image | code | code_image | gpt | seedream        (default image)
#   LIMIT    number of examples; empty = all                   (default all)
cd "$(dirname "$0")"
source ./_common.sh

python "$EDITOR" \
    --input_file "$PROMPT_FILE" \
    --output_dir "$EDITED_DIR" \
    --model_path "$MODEL" \
    --operation "$TASK" \
    --num_workers "$WORKERS" \
    --no_wandb \
    ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} ${EXTRA_EDIT_ARGS[@]+"${EXTRA_EDIT_ARGS[@]}"}

echo "Edits written to ${EDITED_DIR}/ — now run: MODEL=$MODEL TASK=$TASK bash run_eval.sh"
