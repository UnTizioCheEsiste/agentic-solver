"""ASP backend using clingo."""

from __future__ import annotations

import clingo

from agentic_solver.mcp_server.backends.base import SolveResult, ValidationResult


class AspBackend:
    """Validate and solve ASP programs through clingo."""

    name = "asp"

    def __init__(self, models: int = 1) -> None:
        self.models = models

    def validate(self, model: str) -> ValidationResult:
        """Validate ASP syntax by grounding the program."""

        try:
            control = clingo.Control([f"--models={self.models}"])
            control.add("base", [], model)
            control.ground([("base", [])])
        except Exception as exc:
            return ValidationResult(valid=False, status="invalid", message=str(exc))

        return ValidationResult(valid=True, status="valid", message="ASP program is valid.")

    def solve(self, model: str) -> SolveResult:
        """Solve an ASP program with clingo."""

        validation = self.validate(model)
        if not validation.valid:
            return SolveResult(status="error", message=validation.message)

        try:
            control = clingo.Control([f"--models={self.models}"])
            control.add("base", [], model)
            control.ground([("base", [])])
            solutions: list[str] = []
            solve_result = control.solve(
                on_model=lambda answer_set: solutions.append(
                    " ".join(str(symbol) for symbol in answer_set.symbols(shown=True))
                )
            )
        except Exception as exc:
            return SolveResult(status="error", message=str(exc))

        if solve_result.unsatisfiable:
            return SolveResult(status="unsat", message=str(solve_result))
        if solve_result.unknown:
            return SolveResult(status="unknown", message=str(solve_result))

        return SolveResult(status="sat", solution="\n".join(solutions), message=str(solve_result))

