"""Append-only execution logging for pipeline runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


# Pipeline-level solver status values:
# - not_started: the pipeline has not reached solver selection/model execution.
# - selection_failed: the first agent failed before selecting a solver.
# - model_validation_failed: the second agent produced a model that did not validate.
# - model_valid: a model validated successfully, but solving has not run or been logged yet.
# - solver_error: the backend failed while solving or the pipeline failed after selection.
# - unsat: the solver proved that the generated model has no solution.
# - unknown: the solver could not determine satisfiability.
# - solved: the solver found at least one satisfying solution.
SolverStatus = Literal[
    "not_started",
    "selection_failed",
    "model_validation_failed",
    "model_valid",
    "solver_error",
    "unsat",
    "unknown",
    "solved",
]


class ExecutionLog(BaseModel):
    """Strict JSONL record for one end-to-end pipeline execution."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    problem_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    selected_solver: str | None = None
    tool_calls: int = Field(default=0, ge=0)
    repair_attempts: int = Field(default=0, ge=0)
    model_valid: bool | None = None
    solver_status: SolverStatus = "not_started"
    solution_correct: bool | None = None
    execution_time_seconds: float = Field(ge=0.0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


def problem_id_from_path(path: str | Path) -> str:
    """Derive a stable problem id from the problem file name."""

    return Path(path).stem


def record_execution(log: ExecutionLog, log_file: str | Path) -> None:
    """Append an execution log record as one JSON object per line."""

    destination = Path(log_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = log.model_dump_json() + "\n"
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
