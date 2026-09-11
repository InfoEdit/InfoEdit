# Copy to env.sh and fill in your own values, then: source env.sh
#
# This branch reaches every model through an OpenAI-compatible proxy, so the
# whole configuration is two variables — no GCP project, no staging bucket.

# Key issued by the proxy.
export OPENAI_API_KEY=YOUR_PROXY_KEY

# Proxy endpoint, including the /v1 suffix.
export OPENAI_BASE_URL=https://your-proxy.example.com/v1

# Check a model before a long run — reports text / image-input / JSON support:
#   python llm_client.py --probe gemini-3.5-flash
