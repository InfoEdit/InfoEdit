# Shared configuration resolved from environment variables.
# Sourced by run_edit.sh / run_eval.sh — not meant to be run directly.
set -euo pipefail

: "${OPENAI_API_KEY:?set OPENAI_API_KEY — copy env.example.sh to env.sh, fill it in, then: source env.sh}"
: "${OPENAI_BASE_URL:?set OPENAI_BASE_URL — the proxy endpoint, e.g. https://…/v1}"

SOURCE="${SOURCE:-html}"        # html | ppt
TASK="${TASK:-text_expand}"     # text_expand | add | swap_inter | aspect_ratio
PATHWAY="${PATHWAY:-code}"      # code | code_image  (pixel pathways are Vertex-only)
MODEL="${MODEL:?set MODEL, e.g. MODEL=gemini-2.5-flash-image}"
JUDGE="${JUDGE:-gemini-3.1-pro-preview}"
LIMIT="${LIMIT:-}"
WORKERS="${WORKERS:-8}"        # online calls: keep well under the proxy rate limit

case "$SOURCE" in
  html) CODE=html; VERSION="${VERSION:-v17}" ;;
  ppt)  CODE=ppt;  VERSION="${VERSION:-v7}"  ;;
  *) echo "SOURCE must be html or ppt (got '$SOURCE')" >&2; exit 1 ;;
esac

PROMPT_FILE="data/editing_prompts_${CODE}/${VERSION}.jsonl"
EXTRA_EDIT_ARGS=()

case "$PATHWAY" in
  image|gpt|seedream)
    echo "PATHWAY=$PATHWAY is not available on the openai-proxy branch." >&2
    echo "Pixel-level editing returns an image from the model, which the" >&2
    echo "OpenAI chat-completions API does not express; those backends still" >&2
    echo "run on Vertex AI. Use the main branch for them, or PATHWAY=code /" >&2
    echo "code_image here." >&2
    exit 1 ;;
  code|code_image)
    # HTML edits the HTML source; PPT edits the slide via python-pptx.
    if [ "$SOURCE" = "ppt" ]; then
      EDITOR=baselines/code/ppt/edit.py
      # The PPT editor rebuilds each slide from the master deck, so it needs
      # the deck itself plus the id -> slide-index mapping.
      EXTRA_EDIT_ARGS+=(--master_pptx "data/ppt_infographics/${VERSION}/tempates_gallery_0430.pptx"
                        --id_mapping_csv "data/ppt_infographics/${VERSION}/id_mapping.csv")
    else
      EDITOR=baselines/code/html/edit.py
    fi
    EDITED_DIR="edited_${CODE}_infographics_code/${VERSION}/${MODEL}_${PATHWAY}"
    RESULT_TAG="${MODEL}_${PATHWAY}"
    EXTRA_EDIT_ARGS+=(--input_mode "$PATHWAY") ;;
  *) echo "PATHWAY must be code or code_image (got '$PATHWAY')" >&2; exit 1 ;;
esac

RESULT_FILE="eval_results/${RESULT_TAG}/${CODE}_${VERSION}.jsonl"
LIMIT_ARGS=(); [ -n "$LIMIT" ] && LIMIT_ARGS=(--limit "$LIMIT")

echo "------------------------------------------------------------"
echo " source=${SOURCE}(${VERSION})  task=${TASK}  pathway=${PATHWAY}"
echo " model=${MODEL}   judge=${JUDGE}   limit=${LIMIT:-all}
 proxy=${OPENAI_BASE_URL}"
echo " prompts = ${PROMPT_FILE}"
echo " edits   = ${EDITED_DIR}/"
echo " results = ${RESULT_FILE}"
echo "------------------------------------------------------------"
