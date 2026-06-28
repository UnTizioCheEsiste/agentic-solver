"""Prolog backend using pyswip."""

from __future__ import annotations

from pathlib import Path
import re
from tempfile import TemporaryDirectory
from uuid import uuid4

from pyswip import Prolog

from agentic_solver.mcp_server.backends.base import SolveResult, ValidationResult


class PrologBackend:
    """Validate and solve Prolog programs through SWI-Prolog via pyswip."""

    name = "prolog"

    def __init__(
        self,
        query_predicate: str = "solve(X)",
        max_solutions: int = 10,
        time_limit_seconds: int = 10,
    ) -> None:
        self.query_predicate = query_predicate
        self.max_solutions = max_solutions
        self.time_limit_seconds = time_limit_seconds

    def validate(self, model: str) -> ValidationResult:
        """Validate Prolog syntax and the configured query predicate."""

        try:
            predicate_indicator = _predicate_indicator(self.query_predicate)
            predicate_name = predicate_indicator.split("/", maxsplit=1)[0]
            if _contains_forbidden_builtin_definition(model):
                return ValidationResult(
                    valid=False,
                    status="invalid",
                    message=(
                        "Prolog program must not define or call built-in "
                        "predicates as top-level clauses. Put findall/3 inside "
                        "solve(Result) instead."
                    ),
                )
            if _contains_unsafe_reachability_recursion(model):
                return ValidationResult(
                    valid=False,
                    status="invalid",
                    message=(
                        "Recursive reachable/1 rules must be cycle-safe. This "
                        "model appears to recurse over reachable/1 without a "
                        "visited list or member/2 guard; use path/3 with visited "
                        "nodes and expose results through solve(Result)."
                    ),
                )
            if not _contains_predicate_definition(model, predicate_name):
                return ValidationResult(
                    valid=False,
                    status="invalid",
                    message=(
                        "Prolog program must define the backend query predicate: "
                        f"{predicate_indicator}."
                    ),
                )

            prolog, module_name = self._consult(model)
            if not list(
                prolog.query(f"current_predicate({module_name}:{predicate_indicator})")
            ):
                return ValidationResult(
                    valid=False,
                    status="invalid",
                    message=(
                        "Prolog program must define the backend query predicate: "
                        f"{predicate_indicator}."
                    ),
                )
        except Exception as exc:
            return ValidationResult(valid=False, status="invalid", message=str(exc))

        return ValidationResult(valid=True, status="valid", message="Prolog program is valid.")

    def solve(self, model: str) -> SolveResult:
        """Consult the Prolog model and query the configured predicate."""

        try:
            prolog, module_name = self._consult(model)
            solutions = list(
                prolog.query(
                    (
                        f"call_with_time_limit({self.time_limit_seconds}, "
                        f"{module_name}:{self.query_predicate})"
                    ),
                    maxresult=self.max_solutions,
                )
            )
        except Exception as exc:
            return SolveResult(status="error", message=str(exc))

        if not solutions:
            return SolveResult(status="unsat", message="No Prolog solutions found.")

        return SolveResult(status="sat", solution=repr(solutions), message="Prolog query succeeded.")

    def _consult(self, model: str) -> tuple[Prolog, str]:
        prolog = Prolog()
        module_name = f"agentic_solver_{uuid4().hex}"
        with TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "model.pl"
            model_file.write_text(
                f":- module({module_name}, []).\n{model}",
                encoding="utf-8",
            )
            list(
                prolog.query(
                    f"load_files('{model_file.as_posix()}', [syntax_errors(error)])"
                )
            )
        return prolog, module_name


def _predicate_indicator(query_predicate: str) -> str:
    match = re.fullmatch(r"\s*([a-z][A-Za-z0-9_]*)\s*\((.*)\)\s*", query_predicate)
    if match is None:
        raise ValueError(f"Unsupported Prolog query predicate: {query_predicate}")

    name = match.group(1)
    arguments = match.group(2).strip()
    arity = 0 if not arguments else arguments.count(",") + 1
    return f"{name}/{arity}"


def _contains_predicate_definition(model: str, predicate_name: str) -> bool:
    return re.search(rf"(?m)^\s*{re.escape(predicate_name)}\s*\(", model) is not None


def _contains_forbidden_builtin_definition(model: str) -> bool:
    return re.search(r"(?m)^\s*findall\s*\(", model) is not None


def _contains_unsafe_reachability_recursion(model: str) -> bool:
    has_recursive_reachable = re.search(
        r"(?ms)^\s*reachable\s*\([^)]*\)\s*:-.*reachable\s*\(",
        model,
    )
    if has_recursive_reachable is None:
        return False

    cycle_guards = ("Visited", "visited", "member(", "memberchk(", "path(")
    return not any(guard in model for guard in cycle_guards)
