"""Main entrypoint for running the agentic solver pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from agentic_solver.agents import build_solver_model, select_solver
from agentic_solver.agents.solver_selector import (
    GenerateText,
    _build_transformers_generator,
    _read_problem_input,
)
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
    solver_selector_generator: GenerateText | None = None,
    model_builder_generator: GenerateText | None = None,
) -> dict[str, Any]:
    """Run the full pipeline for a problem file."""

    started_at = time.perf_counter()
    selected_solver: str | None = None
    tool_calls = 0
    repair_attempts = 0
    model_valid: bool | None = None
    solver_status = "not_started"

    try:
        _log_progress(f"reading problem file: {problem_file}")
        problem_input = _read_problem_input(problem_file)
        shared_generator = None
        if solver_selector_generator is None and model_builder_generator is None:
            shared_generator = _lazy_shared_generator(model_id)

        _log_progress("selecting solver")
        solver_selection = select_solver(
            problem_input.problem,
            model_id=model_id,
            generator=solver_selector_generator or shared_generator,
        )
        selected_solver = solver_selection["solver"]
        _log_progress(f"selected solver: {selected_solver}")
        _log_progress("building solver model")
        model_build = build_solver_model(
            problem_input.problem,
            selected_solver,
            model_id=model_id,
            generator=model_builder_generator or shared_generator,
        )
        _log_progress("solver model completed")
        tool_calls = model_build["tool_calls"]
        repair_attempts = model_build["repair_attempts"]
        model_valid = bool(model_build["validation"]["valid"])
        solver_status = _pipeline_solver_status(model_build)
        return {
            "solver_selection": solver_selection,
            "model_build": model_build,
        }
    except Exception:
        if selected_solver is None:
            solver_status = "selection_failed"
        else:
            solver_status = "solver_error"
        raise
    finally:
        if log_file is not None:
            record_execution(
                ExecutionLog(
                    problem_id=problem_id_from_path(problem_file),
                    model_name=model_id,
                    selected_solver=selected_solver,
                    tool_calls=tool_calls,
                    repair_attempts=repair_attempts,
                    model_valid=model_valid,
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


def _pipeline_solver_status(model_build: dict[str, Any]) -> str:
    validation = model_build.get("validation")
    if validation is not None and not validation["valid"]:
        return "model_validation_failed"

    solve = model_build.get("solve")
    if solve is None:
        return "solver_error"

    status = solve["status"]
    if status == "sat":
        return "solved"
    if status in {"unsat", "unknown"}:
        return status
    return "solver_error"


def _log_progress(message: str) -> None:
    print(f"[agentic-solver] {message}", file=sys.stderr, flush=True)


def _lazy_shared_generator(model_id: str) -> GenerateText:
    generator: GenerateText | None = None

    def generate(prompt: str) -> str:
        nonlocal generator
        if generator is None:
            _log_progress(f"loading shared model: {model_id}")
            generator = _build_transformers_generator(model_id)
        return generator(prompt)

    return generate


if __name__ == "__main__":
    raise SystemExit(main())
