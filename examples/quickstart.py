"""Runnable example: load .env, build a Router, route a prompt.

Not part of the public API -- the `llm_router` package itself never reads
.env or manages secrets (see docs/architecture.md). This script is a dev
convenience entry point, exactly like training/train.py, so ad hoc local
runs pick up HF_TOKEN from .env the same way the training pipeline does,
avoiding the Hugging Face Hub "unauthenticated requests" rate-limit
warning when downloading the embedding model.

Usage:
    python examples/quickstart.py ["your prompt here"]
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

MAIN_DIR = Path(__file__).resolve().parent.parent
load_dotenv(MAIN_DIR / ".env")

from llm_router import Router  # noqa: E402  (import after load_dotenv on purpose)


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Write a function to reverse a string"

    config_path = MAIN_DIR / "config.yml"
    if not config_path.is_file():
        config_path = MAIN_DIR / "config.example.yml"

    router = Router.from_config(config_path)
    result = router.route(prompt)

    print(f"prompt:     {prompt}")
    print(f"category:   {result.category}")
    print(f"confidence: {result.confidence:.4f}")
    print(f"model_id:   {result.model_id}")
    print(f"reason:     {result.reason}")
    for candidate in result.candidates:
        print(f"  candidate: {candidate.model_id} score={candidate.final_score:.4f}")


if __name__ == "__main__":
    main()
