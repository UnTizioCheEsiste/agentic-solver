import json

from agentic_solver import main as pipeline_main


def test_run_pipeline_connects_selector_to_model_builder_and_logs(monkeypatch, tmp_path) -> None:
    solver_selection = {
        "solver": "minizinc",
        "problem_type": "constraint_satisfaction",
        "confidence": 0.95,
        "reason": "Assignment problem with finite-domain constraints.",
    }
    problem_analysis = {
        "problem_summary": "Return a satisfying model.",
        "entities": [],
        "quantities": [],
        "relations": [],
        "target": "satisfying assignment",
        "assumptions": [],
        "output_interpretation": "Raw output is the answer.",
    }
    model_build = {
        "solver": "minizinc",
        "items": [{"index": 1, "content": "solve satisfy;"}],
        "model": "solve satisfy;",
        "validation": {"valid": True, "status": "valid", "message": "valid"},
        "solve": {"status": "sat", "solution": "ok", "message": "SATISFIED"},
        "tool_calls": 4,
        "repair_attempts": 1,
        "output_contract": "Raw output is the answer.",
        "problem_analysis": problem_analysis,
        "answer_artifact": {
            "problem": "test",
            "solver": "minizinc",
            "problem_analysis": problem_analysis,
            "model": "solve satisfy;",
            "solver_status": "sat",
            "raw_solution": "ok",
            "solver_message": "SATISFIED",
            "output_contract": "Raw output is the answer.",
        },
    }

    def fake_select_solver(problem, *, model_id, generator):
        assert problem
        assert model_id == "test-model"
        assert generator is not None
        return solver_selection

    def fake_build_solver_model(problem, solver, *, model_id, generator):
        assert problem
        assert solver == "minizinc"
        assert model_id == "test-model"
        assert generator is not None
        return model_build

    monkeypatch.setattr(pipeline_main, "select_solver", fake_select_solver)
    monkeypatch.setattr(pipeline_main, "build_solver_model", fake_build_solver_model)

    assert pipeline_main.run_pipeline(
        "problems/example.json",
        model_id="test-model",
        log_file=tmp_path / "executions.jsonl",
    ) == {
        "solver_selection": solver_selection,
        "model_build": model_build,
    }

    log_lines = (tmp_path / "executions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    log_record = json.loads(log_lines[0])
    assert log_record["problem_id"] == "example"
    assert log_record["model_name"] == "test-model"
    assert log_record["selected_solver"] == "minizinc"
    assert log_record["tool_calls"] == 4
    assert log_record["repair_attempts"] == 1
    assert log_record["model_valid"] is True
    assert log_record["solver_status"] == "solved"
    assert log_record["solution_correct"] is None
    assert log_record["execution_time_seconds"] >= 0
    assert log_record["input_tokens"] is None
    assert log_record["output_tokens"] is None


def test_main_prints_pipeline_result(monkeypatch, capsys) -> None:
    expected = {
        "solver_selection": {"solver": "asp"},
        "model_build": {"solve": {"status": "sat"}},
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
