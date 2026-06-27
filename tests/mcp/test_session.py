from agentic_solver.mcp_server.backends.base import SolveResult, ValidationResult
from agentic_solver.mcp_server.session import ModelSession


class FakeBackend:
    name = "minizinc"

    def __init__(self) -> None:
        self.valid = True

    def validate(self, model: str) -> ValidationResult:
        if self.valid:
            return ValidationResult(valid=True, status="valid", message=f"valid: {model}")
        return ValidationResult(valid=False, status="invalid", message="syntax error")

    def solve(self, model: str) -> SolveResult:
        return SolveResult(status="sat", solution=f"solution: {model}", message="solved")


def test_model_session_builds_and_returns_model() -> None:
    session = ModelSession(backends={"minizinc": FakeBackend()})

    clear_response = session.clear_model("minizinc")
    add_response = session.add_item("var 1..3: x;")
    session.add_item("constraint x > 1;")
    model_response = session.get_model()

    assert clear_response.ok is True
    assert add_response.data["index"] == 1
    assert model_response.data["items"] == [
        {"index": 1, "content": "var 1..3: x;"},
        {"index": 2, "content": "constraint x > 1;"},
    ]
    assert model_response.data["model"] == "var 1..3: x;\nconstraint x > 1;"
    assert model_response.data["tool_calls"] == 4
    assert model_response.data["repair_attempts"] == 0


def test_model_session_replace_delete_and_repair_count() -> None:
    backend = FakeBackend()
    session = ModelSession(backends={"minizinc": backend})

    session.clear_model("minizinc")
    session.add_item("broken")
    backend.valid = False
    validation = session.validate_model()
    replacement = session.replace_item(1, "var 1..3: x;")
    deletion = session.delete_item(1)

    assert validation.ok is False
    assert replacement.ok is True
    assert replacement.data["repair_attempts"] == 1
    assert deletion.ok is True
    assert deletion.data["repair_attempts"] == 1
    assert deletion.data["items"] == []


def test_model_session_delete_returns_reindexed_items() -> None:
    session = ModelSession(backends={"minizinc": FakeBackend()})

    session.clear_model("minizinc")
    session.add_item("first")
    session.add_item("second")
    session.add_item("third")
    response = session.delete_item(2)

    assert response.ok is True
    assert response.data["deleted_index"] == 2
    assert response.data["items"] == [
        {"index": 1, "content": "first"},
        {"index": 2, "content": "third"},
    ]
    assert response.data["model"] == "first\nthird"


def test_model_session_requires_solver_before_validation() -> None:
    session = ModelSession(backends={"minizinc": FakeBackend()})

    response = session.validate_model()

    assert response.ok is False
    assert response.message == "No solver selected. Call clear_model first."


def test_model_session_solves_with_selected_backend() -> None:
    session = ModelSession(backends={"minizinc": FakeBackend()})

    session.clear_model("minizinc")
    session.add_item("solve satisfy;")
    response = session.solve_model()

    assert response.ok is True
    assert response.data["solve"]["status"] == "sat"
    assert response.data["solve"]["solution"] == "solution: solve satisfy;"

