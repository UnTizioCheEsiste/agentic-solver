"""Prolog backend using pyswip."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from pyswip import Prolog

from agentic_solver.mcp_server.backends.base import SolveResult, ValidationResult


class PrologBackend:
    """Validate and solve Prolog programs through SWI-Prolog via pyswip."""

    name = "prolog"

    def __init__(self, query_predicate: str = "solve(X)", max_solutions: int = 10) -> None:
        self.query_predicate = query_predicate
        self.max_solutions = max_solutions

    def validate(self, model: str) -> ValidationResult:
        """Validate Prolog syntax by consulting a temporary source file."""

        try:
            self._consult(model)
        except Exception as exc:
            return ValidationResult(valid=False, status="invalid", message=str(exc))

        return ValidationResult(valid=True, status="valid", message="Prolog program is valid.")

    def solve(self, model: str) -> SolveResult:
        """Consult the Prolog model and query the configured predicate."""

        try:
            prolog = self._consult(model)
            solutions = list(prolog.query(self.query_predicate, maxresult=self.max_solutions))
        except Exception as exc:
            return SolveResult(status="error", message=str(exc))

        if not solutions:
            return SolveResult(status="unsat", message="No Prolog solutions found.")

        return SolveResult(status="sat", solution=repr(solutions), message="Prolog query succeeded.")

    def _consult(self, model: str) -> Prolog:
        prolog = Prolog()
        with TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "model.pl"
            model_file.write_text(model, encoding="utf-8")
            list(
                prolog.query(
                    f"load_files('{model_file.as_posix()}', [syntax_errors(error)])"
                )
            )
        return prolog
