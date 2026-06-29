"""MiniZinc backend using the Gecode solver."""

from __future__ import annotations

from datetime import timedelta
import re

import minizinc

from agentic_solver.mcp_server.backends.base import SolveResult, ValidationResult


class MiniZincBackend:
    """Validate and solve MiniZinc models through Gecode."""

    name = "minizinc"

    def __init__(self, solver_tag: str = "gecode", time_limit_seconds: int = 10) -> None:
        self.solver_tag = solver_tag
        self.time_limit = timedelta(seconds=time_limit_seconds)

    def validate(self, model: str) -> ValidationResult:
        """Validate MiniZinc syntax and instance analysis."""

        prevalidation_error = _prevalidate_minizinc_model(model)
        if prevalidation_error is not None:
            return ValidationResult(
                valid=False,
                status="invalid",
                message=prevalidation_error,
            )

        try:
            instance = self._build_instance(model)
            instance.analyse()
        except Exception as exc:
            return ValidationResult(valid=False, status="invalid", message=str(exc))

        return ValidationResult(valid=True, status="valid", message="MiniZinc model is valid.")

    def solve(self, model: str) -> SolveResult:
        """Run the MiniZinc model with Gecode."""

        validation = self.validate(model)
        if not validation.valid:
            return SolveResult(status="error", message=validation.message)

        try:
            result = self._build_instance(model).solve(time_limit=self.time_limit)
        except Exception as exc:
            return SolveResult(status="error", message=str(exc))

        status_text = str(result.status).lower()
        if "unsatisfiable" in status_text:
            return SolveResult(status="unsat", message=str(result.status))
        if "unknown" in status_text:
            return SolveResult(status="unknown", message=str(result.status))

        return SolveResult(status="sat", solution=str(result), message=str(result.status))

    def _build_instance(self, model_text: str) -> minizinc.Instance:
        solver = minizinc.Solver.lookup(self.solver_tag)
        model = minizinc.Model()
        model.add_string(model_text)
        return minizinc.Instance(solver, model)


def _prevalidate_minizinc_model(model: str) -> str | None:
    if re.search(r"\bsolve\s+satisfy\s*:", model):
        return "MiniZinc solve statements must use `solve satisfy;`, not `solve satisfy:`."

    output_match = re.search(r"(?ms)\boutput\s*\[(.*?)\]\s*;", model)
    if output_match is not None:
        output_body = output_match.group(1)
        if '"' not in output_body and "show(" not in output_body:
            return (
                "MiniZinc output must be an array of strings. Use string literals "
                "and show(...) for numeric values."
            )

    for raw_line in model.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", line):
            return (
                "MiniZinc does not allow bare assignment statements after "
                "declarations. Use `constraint ...;` or initialize a parameter "
                "in its declaration."
            )

    return None
