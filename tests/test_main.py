import json

from agentic_solver import main as pipeline_main


def test_run_pipeline_delegates_to_solver_selector(monkeypatch) -> None:
    expected = {
        "solver": "minizinc",
        "problem_type": "constraint_satisfaction",
        "confidence": 0.95,
        "reason": "Assignment problem with finite-domain constraints.",
    }

    def fake_select_solver_from_file(problem_file, *, model_id):
        assert problem_file == "problems/example.json"
        assert model_id == "test-model"
        return expected

    monkeypatch.setattr(
        pipeline_main,
        "select_solver_from_file",
        fake_select_solver_from_file,
    )

    assert pipeline_main.run_pipeline(
        "problems/example.json",
        model_id="test-model",
    ) == expected


def test_main_prints_pipeline_result(monkeypatch, capsys) -> None:
    expected = {
        "solver": "asp",
        "problem_type": "declarative_planning",
        "confidence": 0.8,
        "reason": "Rules and admissible sets.",
    }

    monkeypatch.setattr(
        pipeline_main,
        "run_pipeline",
        lambda problem_file, *, model_id: expected,
    )

    exit_code = pipeline_main.main(["problems/example.json", "--model-id", "test-model"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == expected
