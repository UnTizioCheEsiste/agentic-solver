"""MiniZinc backend using the Gecode solver."""

from __future__ import annotations

from datetime import timedelta

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

