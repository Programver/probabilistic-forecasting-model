#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

show_help() {
  cat <<'EOF'
Usage:
  ./run_project.sh [pipeline|app]

Commands:
  pipeline  Run the offline forecasting pipeline with ./run.sh
  app       Launch the Streamlit dashboard

Examples:
  ./run_project.sh
  ./run_project.sh pipeline
  ./run_project.sh app
EOF
}

if [[ ${1:-pipeline} == "-h" || ${1:-pipeline} == "--help" ]]; then
  show_help
  exit 0
fi

mode="${1:-pipeline}"

case "$mode" in
  pipeline)
    echo "[run_project] Running forecast pipeline..."
    ./run.sh
    ;;
  app)
    echo "[run_project] Launching Streamlit app..."
    if ! command -v streamlit >/dev/null 2>&1; then
      echo "[run_project] Streamlit is not installed. Install app dependencies with:"
      echo "  pip install -r requirements-app.txt"
      exit 1
    fi
    streamlit run app/streamlit_app.py
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    show_help >&2
    exit 1
    ;;
esac
