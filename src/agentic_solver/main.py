"""Main entrypoint for running the agentic solver pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from agentic_solver.agents import select_solver_from_file
from agentic_solver.config import DEFAULT_EXECUTION_LOG, DEFAULT_MODEL_ID
from agentic_solver.execution_logs import (
    ExecutionLog,
    problem_id_from_path,
    record_execution,
)


def run_pipeline(
    problem_file: str | Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    log_file: str | Path | None = DEFAULT_EXECUTION_LOG,
) -> dict[str, Any]:
    """Run the full pipeline for a problem file."""

    # For now, the complete pipeline is the solver selector. The return value is
    # the strict JSON-compatible solver-selection payload.

    started_at = time.perf_counter()
    selected_solver: str | None = None
    solver_status = "not_started"

    try:
        result = select_solver_from_file(problem_file, model_id=model_id)
        selected_solver = result["solver"]
        solver_status = "model_valid"
        return result
    except Exception:
        solver_status = "selection_failed"
        raise
    finally:
        if log_file is not None:
            record_execution(
                ExecutionLog(
                    problem_id=problem_id_from_path(problem_file),
                    model_name=model_id,
                    selected_solver=selected_solver,
                    tool_calls=0,
                    repair_attempts=0,
                    model_valid=None,
                    solver_status=solver_status,
                    solution_correct=None,
                    execution_time_seconds=time.perf_counter() - started_at,
                    input_tokens=None,
                    output_tokens=None,
                ),
                log_file=log_file,
            )


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
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_EXECUTION_LOG),
        help="Path to the JSONL execution log file. Use 'none' to disable logging.",
    )
    args = parser.parse_args(argv)

    log_file = None if args.log_file.lower() == "none" else args.log_file
    result = run_pipeline(args.problem_file, model_id=args.model_id, log_file=log_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
