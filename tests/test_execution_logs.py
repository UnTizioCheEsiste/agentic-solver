import json

from agentic_solver.execution_logs import ExecutionLog, problem_id_from_path, record_execution


def test_problem_id_from_path_uses_file_stem() -> None:
    assert problem_id_from_path("problems/example.json") == "example"


def test_record_execution_appends_jsonl(tmp_path) -> None:
    log_file = tmp_path / "nested" / "executions.jsonl"
    log = ExecutionLog(
        problem_id="example",
        model_name="test-model",
        selected_solver="asp",
        solver_status="solved",
        model_valid=True,
        solution_correct=True,
        execution_time_seconds=1.25,
        input_tokens=10,
        output_tokens=20,
    )

    record_execution(log, log_file)
    record_execution(log, log_file)

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first_record = json.loads(lines[0])
    assert first_record["problem_id"] == "example"
    assert first_record["model_name"] == "test-model"
    assert first_record["selected_solver"] == "asp"
    assert first_record["solver_status"] == "solved"
    assert first_record["model_valid"] is True
    assert first_record["solution_correct"] is True
    assert first_record["execution_time_seconds"] == 1.25
    assert first_record["input_tokens"] == 10
    assert first_record["output_tokens"] == 20
