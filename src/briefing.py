#!/usr/bin/env python3
"""
Generate an AI forecast briefing from the command line (demo / off the scored path).

Unlike ``predict.py`` (which stays strictly offline for scoring), this tool may
call an LLM to polish the grounded narrative when a key is available; otherwise it
prints the fully-offline template briefing. Handy for the demo walkthrough.

Usage:
    python src/briefing.py                          # uses ./data + ./pickle/model.pkl
    python src/briefing.py --data-dir ./data --no-llm
    GEMINI_API_KEY=... python src/briefing.py --output output/briefing.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "insights"))

# python-dotenv is an app-layer extra (requirements-app.txt), not a scored-path
# dependency — so loading a local .env is best-effort. Without it this tool still
# runs on requirements.txt alone and still honours real environment variables; it
# just cannot read a .env file. Never make this import fatal.
try:
    from dotenv import load_dotenv  # noqa: E402
except ImportError:
    pass
else:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import config  # noqa: E402
from forecast import run_forecast  # noqa: E402
from ingest import load_canonical  # noqa: E402
from logging_utils import get_logger  # noqa: E402
from model import load_model  # noqa: E402
from preprocess import prepare  # noqa: E402
from engine import generate_briefing  # noqa: E402

log = get_logger("briefing")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate an AI forecast briefing.")
    p.add_argument("--data-dir", default=str(config.DEFAULT_DATA_DIR))
    p.add_argument("--model", default=str(config.DEFAULT_MODEL_PATH))
    p.add_argument("--output", default=None, help="optional path to save markdown")
    p.add_argument("--no-llm", action="store_true", help="force offline template narrative")
    args = p.parse_args(argv)

    panel = load_canonical(args.data_dir)
    prepared = prepare(panel)
    model = load_model(args.model)
    result = run_forecast(prepared, model)
    brief = generate_briefing(result, prepared, model, use_llm=not args.no_llm)

    print("\n" + "=" * 78)
    print(f"Insight provider: {brief.provider}" + (f" ({brief.model})" if brief.model else ""))
    print("=" * 78 + "\n")
    print(brief.markdown)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(brief.markdown)
        (out.with_suffix(".json")).write_text(json.dumps(brief.to_dict(), indent=2, default=str))
        log.info("saved briefing -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
