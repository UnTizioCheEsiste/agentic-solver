import json

from agentic_solver import main as pipeline_main


def test_run_pipeline_delegates_to_solver_selector_and_logs(monkeypatch, tmp_path) -> None:
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
        log_file=tmp_path / "executions.jsonl",
    ) == expected

    log_lines = (tmp_path / "executions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    log_record = json.loads(log_lines[0])
    assert log_record["problem_id"] == "example"
    assert log_record["model_name"] == "test-model"
    assert log_record["selected_solver"] == "minizinc"
    assert log_record["tool_calls"] == 0
    assert log_record["repair_attempts"] == 0
    assert log_record["model_valid"] is None
    assert log_record["solver_status"] == "model_valid"
    assert log_record["solution_correct"] is None
    assert log_record["execution_time_seconds"] >= 0
    assert log_record["input_tokens"] is None
    assert log_record["output_tokens"] is None


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
        lambda problem_file, *, model_id, log_file: expected,
    )

    exit_code = pipeline_main.main(
        [
            "problems/example.json",
            "--model-id",
            "test-model",
            "--log-file",
            "none",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == expected
