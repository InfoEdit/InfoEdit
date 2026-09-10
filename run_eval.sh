#!/usr/bin/env bash
# Step 2 — score edited infographics with the reflow-aware MLLM judge.
# Reports Edit Compliance (EC), Content Preservation (CP) and Success Rate (SR).
#
#   MODEL=gemini-2.5-flash-image TASK=add bash run_eval.sh
#
# Takes the same environment variables as run_edit.sh, plus:
#   JUDGE    judge model  (default gemini-3.1-pro-preview, as in the paper)
cd "$(dirname "$0")"
source ./_common.sh

python evaluate_edits.py \
    --input_file "$PROMPT_FILE" \
    --edited_dir "$EDITED_DIR" \
    --output_file "$RESULT_FILE" \
    --model_path "$JUDGE" \
    --operation "$TASK" \
    --num_workers "$WORKERS" \
    --use_batch \
    --gcp_project "$GCP_PROJECT" \
    --gcp_location "$GCP_LOCATION" \
    --batch_bucket_uri "$BATCH_BUCKET_URI" \
    --no_wandb \
    --detailed \
    ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}

echo ""
python summarize_eval.py "$VERSION" --model "$RESULT_TAG" --prefix "$CODE" --ops "$TASK"
