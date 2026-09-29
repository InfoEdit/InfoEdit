# Shared configuration resolved from environment variables.
# Sourced by run_edit.sh / run_eval.sh from the repo root — not meant to be run directly.
set -euo pipefail

# vertex: Vertex AI batch prediction (default, every pathway).
# openai: any model behind an OpenAI-compatible proxy, called online — code
#         pathways and the judge only, since pixel editing returns images.
BACKEND="${BACKEND:-vertex}"
case "$BACKEND" in
  vertex)
    : "${GCP_PROJECT:?set GCP_PROJECT — copy scripts/env.example.sh to env.sh, fill it in, then: source env.sh}"
    GCP_LOCATION="${GCP_LOCATION:-global}"
    BATCH_BUCKET_URI="${BATCH_BUCKET_URI:-gs://${GCP_PROJECT}-batch-io}"
    BACKEND_ARGS=(--use_batch --gcp_project "$GCP_PROJECT" --gcp_location "$GCP_LOCATION"
                  --batch_bucket_uri "$BATCH_BUCKET_URI")
    DEFAULT_WORKERS=50 ;;
  openai)
    : "${OPENAI_API_KEY:?set OPENAI_API_KEY — the key your proxy issued (see scripts/env.example.sh)}"
    : "${OPENAI_BASE_URL:?set OPENAI_BASE_URL — the proxy endpoint, e.g. https://…/v1}"
    BACKEND_ARGS=(--backend openai)
    DEFAULT_WORKERS=8 ;;        # online calls: keep well under the proxy rate limit
  *) echo "BACKEND must be vertex or openai (got '$BACKEND')" >&2; exit 1 ;;
esac

SOURCE="${SOURCE:-html}"        # html | ppt
TASK="${TASK:-text_expand}"     # text_expand | add | swap_inter | aspect_ratio
PATHWAY="${PATHWAY:-image}"     # image | code | code_image | gpt | seedream
MODEL="${MODEL:?set MODEL, e.g. MODEL=gemini-2.5-flash-image}"
JUDGE="${JUDGE:-gemini-3.1-pro-preview}"
LIMIT="${LIMIT:-}"
WORKERS="${WORKERS:-$DEFAULT_WORKERS}"

case "$SOURCE" in
  html|ppt) CODE="$SOURCE" ;;
  *) echo "SOURCE must be html or ppt (got '$SOURCE')" >&2; exit 1 ;;
esac

# data/editing_prompts/<source>.jsonl stands for the per-task files <source>.<task>.jsonl.
PROMPT_FILE="data/editing_prompts/${CODE}.jsonl"

# PPT slides are rendered with LibreOffice, which picks fonts through fontconfig. The
# reference images were rendered with the open fonts in fonts/ (see README, "Fonts for
# PPT rendering"); point fontconfig there so edited slides render with the same fonts.
if [ "$CODE" = ppt ] && [ -f fonts/fonts.conf ]; then
  export FONTCONFIG_FILE="$PWD/fonts/fonts.conf"
fi
EXTRA_EDIT_ARGS=()

if [ "$BACKEND" = openai ] && [ "$PATHWAY" != code ] && [ "$PATHWAY" != code_image ]; then
  echo "PATHWAY=$PATHWAY needs a model that returns images, which the OpenAI" >&2
  echo "chat-completions API does not express — use BACKEND=vertex, or" >&2
  echo "PATHWAY=code / code_image with BACKEND=openai." >&2
  exit 1
fi

case "$PATHWAY" in
  image)
    EDITOR=baselines/pixel/gemini/edit.py
    EDITED_DIR="edited_${CODE}_infographics/${MODEL}"
    RESULT_TAG="${MODEL}" ;;
  code|code_image)
    # HTML edits the HTML source; PPT edits the slide via python-pptx.
    if [ "$SOURCE" = "ppt" ]; then
      EDITOR=baselines/code/ppt/edit.py
      # The PPT editor rebuilds each slide from the master deck, so it needs
      # the deck itself plus the id -> slide-index mapping.
      EXTRA_EDIT_ARGS+=(--master_pptx "data/ppt_infographics/master_deck.pptx"
                        --id_mapping_csv "data/ppt_infographics/id_mapping.csv")
    else
      EDITOR=baselines/code/html/edit.py
    fi
    EDITED_DIR="edited_${CODE}_infographics_code/${MODEL}_${PATHWAY}"
    RESULT_TAG="${MODEL}_${PATHWAY}"
    EXTRA_EDIT_ARGS+=(--input_mode "$PATHWAY") ;;
  gpt)
    EDITOR=baselines/pixel/gpt/edit.py
    EDITED_DIR="edited_${CODE}_infographics_gpt/${MODEL}"
    RESULT_TAG="${MODEL}" ;;
  seedream)
    EDITOR=baselines/pixel/seedream/edit.py
    EDITED_DIR="edited_${CODE}_infographics_seedream/${MODEL}"
    RESULT_TAG="${MODEL}" ;;
  *) echo "PATHWAY must be image|code|code_image|gpt|seedream (got '$PATHWAY')" >&2; exit 1 ;;
esac

RESULT_FILE="eval_results/${RESULT_TAG}/${CODE}.jsonl"
LIMIT_ARGS=(); [ -n "$LIMIT" ] && LIMIT_ARGS=(--limit "$LIMIT")

echo "------------------------------------------------------------"
echo " source=${SOURCE}  task=${TASK}  pathway=${PATHWAY}"
echo " model=${MODEL}   judge=${JUDGE}   limit=${LIMIT:-all}   backend=${BACKEND}"
echo " prompts = ${PROMPT_FILE}"
echo " edits   = ${EDITED_DIR}/"
echo " results = ${RESULT_FILE}"
echo "------------------------------------------------------------"
