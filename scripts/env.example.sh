# Copy to env.sh in the repo root and fill in your own values, then: source env.sh
#
# Vertex AI project that will run Gemini batch prediction jobs.
export GCP_PROJECT=YOUR_GCP_PROJECT_ID

# Vertex AI endpoint location. "global" is recommended; some models are
# region-restricted, in which case use e.g. us-central1.
export GCP_LOCATION=global

# GCS bucket used to stage batch input/output. Must be a SINGLE region
# (us-central1), not the multi-region "us".
#   gcloud storage buckets create gs://${GCP_PROJECT}-batch-io --location=us-central1
export BATCH_BUCKET_URI=gs://${GCP_PROJECT}-batch-io

# Optional, per backend:
#   GPT-Image-2 (baselines/pixel/gpt)
# export OPENAI_API_KEY=sk-...
#   BACKEND=openai — code-level editing and judging through an OpenAI-compatible
#   proxy instead of Vertex AI (no GCP project needed). Check a model first with
#   python utils/llm_client.py --probe MODEL
# export BACKEND=openai
# export OPENAI_API_KEY=YOUR_PROXY_KEY
# export OPENAI_BASE_URL=https://your-proxy.example.com/v1
#   Seedream via Volcengine Ark (baselines/pixel/seedream)
# export ARK_API_KEY=...
