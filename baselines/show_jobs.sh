#!/usr/bin/env bash
# Show Vertex AI batch prediction jobs in the current gcloud account/project.
#
# Usage:
#   bash show_jobs.sh              # only RUNNING / PENDING / QUEUED
#   bash show_jobs.sh --all        # include SUCCEEDED / FAILED / CANCELLED
#   bash show_jobs.sh --raw        # also print raw API response (debug)
#
# Env overrides (in priority order):
#   GCP_PROJECT     project to query
#   PROJECT         same as GCP_PROJECT (compat with README curl examples)
#   GCP_LOCATIONS   space-separated regions to scan
#                   (default: "global us-west1 us-central1")

set -u

PROJECT_ID=${GCP_PROJECT:-${PROJECT:-$(gcloud config get-value project 2>/dev/null)}}
LOCATIONS=${GCP_LOCATIONS:-"global us-west1 us-central1"}
ACCOUNT=$(gcloud config get-value account 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    echo "Error: no project set. Run 'gcloud config set project <id>' or 'export GCP_PROJECT=<id>'." >&2
    exit 1
fi

STATE_FILTER='JOB_STATE_RUNNING|JOB_STATE_PENDING|JOB_STATE_QUEUED'
MODE_LABEL="active only"
SHOW_RAW=0
for arg in "$@"; do
    case "$arg" in
        --all) STATE_FILTER='.*'; MODE_LABEL="all states" ;;
        --raw) SHOW_RAW=1 ;;
        -h|--help)
            sed -n '2,15p' "$0"; exit 0 ;;
    esac
done

TOKEN=$(gcloud auth print-access-token) || {
    echo "Error: failed to get access token. Check 'gcloud auth list'." >&2
    exit 1
}

echo "================================================================"
echo " Account:   ${ACCOUNT:-<unknown>}"
echo " Project:   $PROJECT_ID"
echo " Locations: $LOCATIONS"
echo " Filter:    $MODE_LABEL"
echo "================================================================"

# Helper: safely turn a possibly-missing/string/number field into a number.
# Vertex AI returns int64 fields as strings in JSON.
JQ_FILTER=$(cat <<'EOF'
def num(x): (x // 0) | tonumber? // 0;

.batchPredictionJobs[]?
| select(.state | test($states))
| num(.completionStats.successfulCount) as $ok
| num(.completionStats.failedCount)     as $fail
| num(.completionStats.incompleteCount) as $todo
| ($ok + $fail) as $done
| ($done + $todo) as $total
| (if $total > 0 then (($done * 100 / $total) | floor) else 0 end) as $pct
| "  [\(.state | sub("JOB_STATE_"; ""))] \(.displayName // "(no name)")\n" +
  "    job_id:   \(.name | split("/") | last)\n" +
  "    model:    \(.model // "-" | split("/") | last)\n" +
  "    progress: \($done)/\($total)  (\($pct)%)   ok=\($ok)  fail=\($fail)  todo=\($todo)\n" +
  "    created:  \(.createTime // "-")\n"
EOF
)

found_total=0
for LOC in $LOCATIONS; do
    URL="https://aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOC}/batchPredictionJobs"
    echo
    echo "[location=$LOC]"

    response=$(curl -sS -H "Authorization: Bearer $TOKEN" "$URL")
    curl_exit=$?

    if [ $curl_exit -ne 0 ]; then
        echo "  (curl failed, exit=$curl_exit)"
        continue
    fi

    # API error?
    err=$(echo "$response" | jq -r '.error.message // empty')
    if [ -n "$err" ]; then
        echo "  API error: $err"
        continue
    fi

    if [ "$SHOW_RAW" = "1" ]; then
        echo "--- raw response ---"
        echo "$response" | jq .
        echo "--- end raw ---"
    fi

    # Total count of jobs returned by API (any state)
    total_returned=$(echo "$response" | jq '.batchPredictionJobs | length // 0')

    # Filtered output
    output=$(echo "$response" | jq -r --arg states "$STATE_FILTER" "$JQ_FILTER")

    if [ -n "$output" ]; then
        echo "$output"
        found_total=$((found_total + 1))
    else
        echo "  (no jobs match filter; API returned $total_returned job(s) total in this location)"
    fi
done

echo
if [ "$found_total" = "0" ]; then
    echo "No matching jobs found across any location."
    echo "Tips:"
    echo "  - bash show_jobs.sh --all     # include finished jobs"
    echo "  - bash show_jobs.sh --raw     # see what the API actually returns"
    echo "  - GCP_LOCATIONS='global' bash show_jobs.sh    # narrow down"
fi
