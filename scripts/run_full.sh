#!/usr/bin/env bash
# Final-submission run: official documents + open LLM + evaluation + sample outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${RCA_OLLAMA_MODEL:-granite3.3:2b}"
export RCA_OLLAMA_MODEL="$MODEL"

command -v ollama >/dev/null || { echo "Install Ollama first: https://ollama.com/download"; exit 1; }
curl -fsS http://localhost:11434/api/tags >/dev/null || {
  echo "Ollama is not running. Start the app, or run: ollama serve"; exit 1; }

python -m pip install -q -e ".[laptop,dev]"
ollama pull "$MODEL"
python scripts/fetch_official.py || true
rm -rf .rca/laptop
rca ingest -p laptop
rca evaluate -p laptop --judge
python scripts/make_samples.py laptop
rca audit-verify -p laptop
python scripts/build_docs.py 2>/dev/null || echo "PDFs not rebuilt (needs markdown and playwright). The Markdown report is current."
echo "Done. Read docs/evaluation/evaluation_report.md and sample_outputs/."
