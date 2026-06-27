"""Common backend contracts for symbolic solvers."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


SolverName = Literal["asp", "prolog", "minizinc"]
SolverStatus = Literal["valid", "invalid", "error", "sat", "unsat", "unknown"]


class ValidationResult(BaseModel):
    """Result returned by backend model validation."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    status: SolverStatus
    message: str = ""


class SolveResult(BaseModel):
    """Result returned by backend model solving."""

    model_config = ConfigDict(extra="forbid")

    status: SolverStatus
    solution: str | None = None
    message: str = ""


class SolverBackend(Protocol):
    """Interface implemented by all concrete solver backends."""

    name: SolverName

    def validate(self, model: str) -> ValidationResult:
        """Validate solver-specific model text."""

    def solve(self, model: str) -> SolveResult:
        """Solve solver-specific model text."""

