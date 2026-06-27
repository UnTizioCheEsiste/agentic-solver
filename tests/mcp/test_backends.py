def test_minizinc_backend_validates_and_solves_with_gecode() -> None:
    from agentic_solver.mcp_server.backends.minizinc import MiniZincBackend

    model = """
var 1..3: x;
constraint x = 2;
solve satisfy;
output [show(x)];
"""
    backend = MiniZincBackend(solver_tag="gecode", time_limit_seconds=5)

    validation = backend.validate(model)
    result = backend.solve(model)

    assert validation.valid is True
    assert result.status == "sat"
    assert result.solution == "2"


def test_minizinc_backend_reports_invalid_model() -> None:
    from agentic_solver.mcp_server.backends.minizinc import MiniZincBackend

    backend = MiniZincBackend(solver_tag="gecode", time_limit_seconds=5)

    validation = backend.validate("var 1..3 x; solve satisfy;")

    assert validation.valid is False
    assert validation.status == "invalid"
    assert validation.message


def test_asp_backend_validates_and_solves_with_clingo() -> None:
    from agentic_solver.mcp_server.backends.asp import AspBackend

    model = """
choose(1).
#show choose/1.
"""
    backend = AspBackend(models=1)

    validation = backend.validate(model)
    result = backend.solve(model)

    assert validation.valid is True
    assert result.status == "sat"
    assert result.solution == "choose(1)"


def test_asp_backend_reports_invalid_model() -> None:
    from agentic_solver.mcp_server.backends.asp import AspBackend

    backend = AspBackend(models=1)

    validation = backend.validate("a(")

    assert validation.valid is False
    assert validation.status == "invalid"
    assert validation.message


def test_prolog_backend_validates_and_solves_with_pyswip() -> None:
    from agentic_solver.mcp_server.backends.prolog import PrologBackend

    backend = PrologBackend(query_predicate="solve(X)")

    validation = backend.validate("solve(2).")
    result = backend.solve("solve(2).")

    assert validation.valid is True
    assert result.status == "sat"
    assert result.solution == "[{'X': 2}]"


def test_prolog_backend_reports_invalid_model() -> None:
    from agentic_solver.mcp_server.backends.prolog import PrologBackend

    backend = PrologBackend(query_predicate="solve(X)")

    validation = backend.validate("solve(")

    assert validation.valid is False
    assert validation.status == "invalid"
    assert validation.message
