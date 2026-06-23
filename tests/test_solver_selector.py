import json

import pytest

from agentic_solver.agents.solver_selector import (
    ProblemInput,
    SolverSelection,
    parse_solver_selection,
    select_solver_from_file,
)


def test_select_solver_from_example_file() -> None:
    result = select_solver_from_file(
        "problems/example.json",
        generator=lambda _prompt: json.dumps(
            {
                "solver": "minizinc",
                "problem_type": "constraint_satisfaction",
                "confidence": 0.95,
                "reason": "Il problema contiene variabili finite e vincoli di assegnamento.",
            }
        ),
    )

    assert result == {
        "solver": "minizinc",
        "problem_type": "constraint_satisfaction",
        "confidence": 0.95,
        "reason": "Il problema contiene variabili finite e vincoli di assegnamento.",
    }


def test_problem_input_requires_problem_string(tmp_path) -> None:
    problem_file = tmp_path / "missing_problem.json"
    problem_file.write_text(json.dumps({"text": "test"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Problem file must contain"):
        select_solver_from_file(
            problem_file,
            generator=lambda _prompt: "{}",
        )

    assert ProblemInput.model_validate({"problem": "test"}).problem == "test"


def test_malformed_problem_json_raises(tmp_path) -> None:
    problem_file = tmp_path / "bad.json"
    problem_file.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Problem file is not valid JSON"):
        select_solver_from_file(problem_file, generator=lambda _prompt: "{}")


def test_model_output_must_be_json() -> None:
    with pytest.raises(ValueError, match="Model output is not valid JSON"):
        parse_solver_selection("not json")


def test_model_output_rejects_unknown_solver() -> None:
    raw_output = json.dumps(
        {
            "solver": "z3",
            "problem_type": "constraint_satisfaction",
            "confidence": 0.7,
            "reason": "Invalid solver.",
        }
    )

    with pytest.raises(ValueError, match="SolverSelection schema"):
        parse_solver_selection(raw_output)


def test_model_output_rejects_extra_fields() -> None:
    raw_output = json.dumps(
        {
            "solver": "minizinc",
            "problem_type": "constraint_satisfaction",
            "confidence": 0.95,
            "reason": "Valid selection.",
            "debug": "not allowed",
        }
    )

    with pytest.raises(ValueError, match="SolverSelection schema"):
        parse_solver_selection(raw_output)


def test_solver_selection_schema_accepts_expected_shape() -> None:
    selection = SolverSelection.model_validate(
        {
            "solver": "asp",
            "problem_type": "declarative_planning",
            "confidence": 0.8,
            "reason": "Il problema parla di regole e insiemi ammissibili.",
        }
    )

    assert selection.model_dump() == {
        "solver": "asp",
        "problem_type": "declarative_planning",
        "confidence": 0.8,
        "reason": "Il problema parla di regole e insiemi ammissibili.",
    }
