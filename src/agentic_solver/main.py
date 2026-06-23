"""Main entrypoint for running the agentic solver pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from agentic_solver.agents import select_solver_from_file
from agentic_solver.config import DEFAULT_MODEL_ID


def run_pipeline(
    problem_file: str | Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    """Run the full pipeline for a problem file."""

    # For now, the complete pipeline is the solver selector. The return value is
    # the strict JSON-compatible solver-selection payload.

    return select_solver_from_file(problem_file, model_id=model_id)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline from the command line and print JSON to stdout."""

    parser = argparse.ArgumentParser(
        description="Run the agentic solver pipeline for a problem JSON file."
    )
    parser.add_argument(
        "problem_file",
        help="Path to a JSON file containing the 'problem' field.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id used by the solver selector.",
    )
    args = parser.parse_args(argv)

    result = run_pipeline(args.problem_file, model_id=args.model_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
