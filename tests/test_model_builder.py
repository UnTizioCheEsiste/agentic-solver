import json

import pytest

from agentic_solver.agents.model_builder import (
    build_solver_model,
    parse_model_build_plan,
    parse_model_repair_plan,
)
from agentic_solver.mcp_server.backends.base import SolveResult, ValidationResult
from agentic_solver.mcp_server.session import ModelSession


class FakeBackend:
    name = "minizinc"

    def __init__(self) -> None:
        self.validation_calls = 0

    def validate(self, model: str) -> ValidationResult:
        self.validation_calls += 1
        if "broken" in model:
            return ValidationResult(valid=False, status="invalid", message="syntax error")
        return ValidationResult(valid=True, status="valid", message="valid")

    def solve(self, model: str) -> SolveResult:
        return SolveResult(status="sat", solution="2", message="SATISFIED")


def test_build_solver_model_returns_structured_answer_artifact() -> None:
    backend = FakeBackend()
    session = ModelSession(backends={"minizinc": backend})
    plan = {
        "items": [
            "var 1..3: x;",
            "constraint x = 2;",
            "solve satisfy;",
            "output [show(x)];",
        ],
        "output_contract": "The raw solution is the value assigned to x.",
    }

    result = build_solver_model(
        "Find x.",
        "minizinc",
        generator=lambda _prompt: json.dumps(plan),
        session=session,
    )

    assert result["solver"] == "minizinc"
    assert result["validation"]["valid"] is True
    assert result["solve"]["status"] == "sat"
    assert result["answer_artifact"] == {
        "problem": "Find x.",
        "solver": "minizinc",
        "model": "var 1..3: x;\nconstraint x = 2;\nsolve satisfy;\noutput [show(x)];",
        "solver_status": "sat",
        "raw_solution": "2",
        "solver_message": "SATISFIED",
        "output_contract": "The raw solution is the value assigned to x.",
    }


def test_build_solver_model_repairs_after_validation_failure() -> None:
    backend = FakeBackend()
    session = ModelSession(backends={"minizinc": backend})
    responses = iter(
        [
            json.dumps(
                {
                    "items": ["broken"],
                    "output_contract": "The raw solution is the final value.",
                }
            ),
            json.dumps(
                {
                    "actions": [
                        {
                            "tool": "replace_item",
                            "index": 1,
                            "content": "var 1..3: x; constraint x = 2; solve satisfy;",
                        }
                    ]
                }
            ),
        ]
    )

    result = build_solver_model(
        "Find x.",
        "minizinc",
        generator=lambda _prompt: next(responses),
        session=session,
    )

    assert result["validation"]["valid"] is True
    assert result["repair_attempts"] == 1
    assert result["items"] == [
        {
            "index": 1,
            "content": "var 1..3: x; constraint x = 2; solve satisfy;",
        }
    ]


def test_parse_model_build_plan_rejects_extra_fields() -> None:
    with pytest.raises(ValueError, match="ModelBuildPlan schema"):
        parse_model_build_plan(
            json.dumps(
                {
                    "items": ["solve satisfy;"],
                    "output_contract": "Raw output is the solution.",
                    "debug": "not allowed",
                }
            )
        )


def test_parse_model_repair_plan_rejects_unknown_tool() -> None:
    with pytest.raises(ValueError, match="ModelRepairPlan schema"):
        parse_model_repair_plan(
            json.dumps({"actions": [{"tool": "clear_model", "content": ""}]})
        )
