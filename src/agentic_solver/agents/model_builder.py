"""Build solver-specific models using the fixed MCP tool interface."""

from __future__ import annotations

import json
import sys
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agentic_solver.agents.solver_selector import GenerateText, SolverName
from agentic_solver.config import DEFAULT_MODEL_ID
from agentic_solver.mcp_server.session import ModelSession


class ModelBuildPlan(BaseModel):
    """Initial model construction plan returned by the model-building agent."""

    model_config = ConfigDict(extra="forbid")

    items: list[str] = Field(min_length=1)
    output_contract: str = Field(min_length=1)


class Quantity(BaseModel):
    """A numeric or symbolic quantity extracted from the problem."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    value: str | int | float | None = None
    unit: str | None = None
    description: str = Field(min_length=1)

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value(cls, value: Any) -> Any:
        if value is None or isinstance(value, str | int | float):
            return value
        return json.dumps(value, ensure_ascii=False)


class ProblemAnalysis(BaseModel):
    """Structured semantic analysis used before generating solver code."""

    model_config = ConfigDict(extra="forbid")

    problem_summary: str = Field(min_length=1)
    entities: list[str] = Field(default_factory=list)
    quantities: list[Quantity] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    target: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    output_interpretation: str = Field(
        default="Interpret the raw solver output as the value of the requested target.",
        min_length=1,
    )

    @field_validator("output_interpretation", mode="before")
    @classmethod
    def default_output_interpretation(cls, value: Any) -> str:
        if value is None or (isinstance(value, str) and not value.strip()):
            return "Interpret the raw solver output as the value of the requested target."
        return value


class RepairAction(BaseModel):
    """One MCP-compatible repair action requested after validation failure."""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(pattern="^(add_item|replace_item|delete_item)$")
    content: str | None = None
    index: int | None = Field(default=None, ge=1)


class ModelRepairPlan(BaseModel):
    """Repair actions returned by the model-building agent."""

    model_config = ConfigDict(extra="forbid")

    actions: list[RepairAction] = Field(default_factory=list)


def build_solver_model(
    problem: str,
    solver: SolverName,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    generator: GenerateText | None = None,
    session: ModelSession | None = None,
    max_repair_attempts: int = 3,
) -> dict[str, Any]:
    """Generate, validate, repair, and solve a model for the selected solver."""

    model_session = session or ModelSession()
    text_generator = generator or _build_transformers_generator(model_id)

    _log_progress("requesting structured problem analysis")
    analysis = _generate_with_schema_repair(
        text_generator=text_generator,
        initial_prompt=build_problem_analysis_prompt(problem, solver),
        parser=parse_problem_analysis,
        payload_name="ProblemAnalysis",
    )

    _log_progress("requesting initial solver model")
    initial_plan = _generate_with_schema_repair(
        text_generator=text_generator,
        initial_prompt=build_model_build_prompt(problem, solver, analysis),
        parser=parse_model_build_plan,
        payload_name="ModelBuildPlan",
    )

    model_session.clear_model(solver)
    for item in initial_plan.items:
        model_session.add_item(item)
    _log_progress(f"added {len(initial_plan.items)} model item(s)")

    repair_rounds = 0
    solve_response = None

    while True:
        validation_response = model_session.validate_model()
        _log_progress(f"validation status: {validation_response.ok}")
        if not validation_response.ok:
            if repair_rounds >= max_repair_attempts:
                break
            repair_rounds += 1
            _repair_model(
                model_session=model_session,
                text_generator=text_generator,
                problem=problem,
                solver=solver,
                model_state=validation_response.data,
                failure_stage="validation",
                failure_message=validation_response.message,
            )
            continue

        solve_response = model_session.solve_model()
        solve_payload = solve_response.data["solve"]
        _log_progress(f"solve status: {solve_payload['status']}")
        if solve_payload["status"] not in {"error", "unsat"} or repair_rounds >= max_repair_attempts:
            break

        repair_rounds += 1
        failure_message = solve_response.message
        if solve_payload["status"] == "unsat":
            failure_message = (
                "The generated model is valid but unsatisfiable. This usually "
                "means one or more constraints encode the problem incorrectly "
                "or mix incompatible arithmetic units."
            )
        _repair_model(
            model_session=model_session,
            text_generator=text_generator,
            problem=problem,
            solver=solver,
            model_state=solve_response.data,
            failure_stage="solve",
            failure_message=_summarize_failure(failure_message),
        )

    solve_payload = solve_response.data["solve"] if solve_response is not None else None
    validation_payload = validation_response.data.get("validation")

    return {
        "solver": solver,
        "items": model_session.get_model().data["items"],
        "model": model_session.model_text,
        "validation": validation_payload,
        "solve": solve_payload,
        "tool_calls": model_session.tool_calls,
        "repair_attempts": repair_rounds,
        "output_contract": initial_plan.output_contract,
        "problem_analysis": analysis.model_dump(),
        "answer_artifact": {
            "problem": problem,
            "solver": solver,
            "problem_analysis": analysis.model_dump(),
            "model": model_session.model_text,
            "solver_status": solve_payload["status"] if solve_payload else "not_solved",
            "raw_solution": solve_payload["solution"] if solve_payload else None,
            "solver_message": solve_payload["message"] if solve_payload else validation_response.message,
            "output_contract": initial_plan.output_contract,
        },
    }


def build_problem_analysis_prompt(problem: str, solver: SolverName) -> str:
    """Build the prompt for the semantic analysis phase."""

    return (
        "You are analyzing a natural-language problem before any solver code is written.\n"
        f"The selected solver is: {solver}.\n\n"
        "Return only valid JSON, without Markdown or extra text. The JSON must "
        "use JSON double quotes for every string and must escape any quotes "
        "inside string values. Do not include comments or trailing commas. "
        "contain exactly these fields:\n"
        "- problem_summary: concise restatement of the task.\n"
        "- entities: array of relevant people, objects, nodes, jobs, tasks, etc.\n"
        "- quantities: array of objects with exactly these fields: name, value, "
        "unit, description. Use null for unknown value or unit.\n"
        "- relations: array of mathematical, logical, graph, temporal, or set "
        "relations that must hold.\n"
        "- target: the exact unknown or output requested by the problem.\n"
        "- assumptions: array of necessary assumptions, including unit conversions "
        "or whether duplicates should be removed.\n"
        "- output_interpretation: how a later answer agent should interpret the raw solver output.\n\n"
        "Do not generate solver code in this phase. Focus on meaning, units, "
        "relations, and the requested target.\n\n"
        "Example quantity object: "
        '{"name": "rate", "value": 12, "unit": "dollars/hour", '
        '"description": "Hourly babysitting rate"}.\n\n'
        f"Problem:\n{problem}"
    )


def build_model_build_prompt(
    problem: str,
    solver: SolverName,
    analysis: ProblemAnalysis,
) -> str:
    """Build the prompt for the model-building agent."""

    return (
        "You are the second agent in a deterministic symbolic-solving pipeline.\n"
        "Your task is to translate the structured problem analysis into code for "
        f"the selected solver: {solver}.\n\n"
        "Return only valid JSON, without Markdown or extra text. The JSON must "
        "use JSON double quotes for every string and must escape any quotes "
        "inside string values. Do not include comments or trailing commas. "
        "contain exactly these fields:\n"
        "- items: a non-empty array of solver-code strings. Each item should be "
        "a coherent replaceable block.\n"
        "- output_contract: a short description of how the raw solver output "
        "should be interpreted by a later natural-language answer agent.\n\n"
        "The solver code must be directly executable by the selected backend. "
        "Use explicit output directives when the solver supports them.\n\n"
        f"{_solver_specific_instructions(solver)}\n\n"
        f"Problem:\n{problem}\n\n"
        f"Structured problem analysis:\n{analysis.model_dump_json()}"
    )


def build_model_repair_prompt(
    *,
    problem: str,
    solver: SolverName,
    model_state: dict[str, Any],
    failure_stage: str,
    failure_message: str,
) -> str:
    """Build the prompt used after validation or solve reports an error."""

    return (
        "You are repairing solver code through fixed MCP tools.\n"
        f"Selected solver: {solver}\n\n"
        f"{_solver_specific_instructions(solver)}\n\n"
        "Return only valid JSON with exactly one field: actions. Each action "
        "must be one of:\n"
        '- {"tool": "replace_item", "index": 1, "content": "..."}\n'
        '- {"tool": "delete_item", "index": 1}\n'
        '- {"tool": "add_item", "content": "..."}\n\n'
        "Use the current 1-based item indexes shown below. After delete, the "
        "server will reindex remaining items.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Failure stage:\n{failure_stage}\n\n"
        f"Failure message:\n{failure_message}\n\n"
        f"{_failure_specific_repair_instructions(solver, failure_message)}\n\n"
        f"Current model state:\n{json.dumps(model_state, ensure_ascii=False)}"
    )


def parse_problem_analysis(raw_output: str) -> ProblemAnalysis:
    """Parse and validate the structured semantic analysis payload."""

    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError("Problem analysis output is not valid JSON.") from exc

    try:
        return ProblemAnalysis.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Problem analysis output does not match ProblemAnalysis schema.") from exc


def parse_model_build_plan(raw_output: str) -> ModelBuildPlan:
    """Parse and validate the initial model-building JSON payload."""

    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError("Model builder output is not valid JSON.") from exc

    try:
        return ModelBuildPlan.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Model builder output does not match ModelBuildPlan schema.") from exc


def parse_model_repair_plan(raw_output: str) -> ModelRepairPlan:
    """Parse and validate a repair JSON payload."""

    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError("Model repair output is not valid JSON.") from exc

    try:
        return ModelRepairPlan.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Model repair output does not match ModelRepairPlan schema.") from exc


def build_schema_repair_prompt(
    *,
    payload_name: str,
    invalid_output: str,
    error: str,
) -> str:
    """Build a retry prompt for malformed JSON or schema-invalid payloads."""

    return (
        f"Your previous {payload_name} response was invalid.\n\n"
        "Return only corrected valid JSON. Do not add Markdown, explanations, "
        "comments, or code fences. Use JSON double quotes for every string, "
        "escape quotes inside strings, and do not use trailing commas.\n\n"
        f"Validation error:\n{error}\n\n"
        f"Invalid previous output:\n{invalid_output}"
    )


def _generate_with_schema_repair(
    *,
    text_generator: GenerateText,
    initial_prompt: str,
    parser: Any,
    payload_name: str,
    max_attempts: int = 2,
) -> Any:
    raw_output = text_generator(initial_prompt)
    for attempt in range(max_attempts + 1):
        try:
            return parser(raw_output)
        except ValueError as exc:
            if attempt >= max_attempts:
                raise
            raw_output = text_generator(
                build_schema_repair_prompt(
                    payload_name=payload_name,
                    invalid_output=raw_output,
                    error=str(exc),
                )
            )

    raise ValueError(f"{payload_name} output did not validate.")


def _apply_repair_plan(session: ModelSession, repair_plan: ModelRepairPlan) -> None:
    for action in repair_plan.actions:
        if action.tool == "add_item":
            if action.content is None:
                raise ValueError("add_item repair action requires content.")
            session.add_item(action.content)
        elif action.tool == "replace_item":
            if action.index is None or action.content is None:
                raise ValueError("replace_item repair action requires index and content.")
            session.replace_item(action.index, action.content)
        elif action.tool == "delete_item":
            if action.index is None:
                raise ValueError("delete_item repair action requires index.")
            session.delete_item(action.index)


def _repair_model(
    *,
    model_session: ModelSession,
    text_generator: GenerateText,
    problem: str,
    solver: SolverName,
    model_state: dict[str, Any],
    failure_stage: str,
    failure_message: str,
) -> None:
    repair_plan = parse_model_repair_plan(
        text_generator(
            build_model_repair_prompt(
                problem=problem,
                solver=solver,
                model_state=model_state,
                failure_stage=failure_stage,
                failure_message=failure_message,
            )
        )
    )
    _apply_repair_plan(model_session, repair_plan)


def _solver_specific_instructions(solver: SolverName) -> str:
    if solver == "prolog":
        return (
            "Prolog backend requirements:\n"
            "- Do not write top-level queries as program clauses.\n"
            "- Do not write clauses for built-in predicates such as findall/3.\n"
            "- The program must define solve(Result). The backend will run "
            "solve(X) and use X as the raw solution.\n"
            "- To return a list, write a rule like: "
            "solve(Result) :- findall(X, reachable(X), Result).\n"
            "- For arithmetic word problems, use `is` to evaluate arithmetic. "
            "Do not use `=` for arithmetic evaluation. Example: "
            "solve(Result) :- Result is 12 * 50 / 60.\n"
            "- Recursive graph programs must terminate on cyclic graphs. Use a "
            "visited list or another cycle-safe strategy, and prefer setof/3 "
            "when duplicate answers should be removed."
        )
    if solver == "asp":
        return (
            "ASP backend requirements:\n"
            "- Include #show directives for exactly the atoms that should be "
            "visible in the raw solution."
        )
    return (
        "MiniZinc backend requirements:\n"
        "- MiniZinc statements end with semicolons. Write `solve satisfy;`, "
        "never `solve satisfy:`.\n"
        "- Variable declarations look like `var 0..100: x;` or `int: x = 48;`.\n"
        "- Relationships between variables must be constraints, for example "
        "`constraint april_clips = 48;` and `constraint 2 * may_clips = april_clips;`.\n"
        "- Do not write bare assignments like `april_clips = 48;` after a "
        "variable declaration.\n"
        "- Avoid `/` for integer division on decision variables. Use `div` only "
        "for integer expressions, or prefer an equivalent multiplication "
        "constraint such as `constraint 2 * may_clips = april_clips;`.\n"
        "- The output item must be an array of strings. Use string literals and "
        "`show(...)`, for example: "
        '`output ["april=", show(april_clips), "\\nmay=", show(may_clips)];`.\n'
        "- Include the target identified by the structured analysis in the output."
    )


def _build_transformers_generator(model_id: str) -> GenerateText:
    from agentic_solver.agents.solver_selector import _build_transformers_generator

    return _build_transformers_generator(model_id)


def _log_progress(message: str) -> None:
    print(f"[model-builder] {message}", file=sys.stderr, flush=True)


def _summarize_failure(message: str) -> str:
    if "resource_error(stack)" in message:
        return (
            "Prolog stack overflow: recursive rules did not terminate, likely "
            "because the graph has a cycle and the model does not track visited nodes."
        )
    if "time_limit_exceeded" in message:
        return (
            "Prolog time limit exceeded: recursive rules likely did not terminate. "
            "Use a cycle-safe formulation with visited nodes."
        )
    return message


def _failure_specific_repair_instructions(solver: SolverName, failure_message: str) -> str:
    if solver == "prolog" and (
        "stack overflow" in failure_message.lower()
        or "time limit" in failure_message.lower()
    ):
        return "Repair requirement: make recursive Prolog rules terminate on cyclic data."
    if solver == "minizinc":
        lower_message = failure_message.lower()
        if "unsatisfiable" in lower_message or "incompatible arithmetic units" in lower_message:
            return (
                "Repair requirement: inspect arithmetic units and constraints. "
                "Do not combine truncated integer division with fractional "
                "division constraints. For rates over minutes/hours, use scaled "
                "integer units such as cents and minutes, or an equivalent exact "
                "integer equation."
            )
        if "unexpected ':'" in lower_message or "solve satisfy:" in lower_message:
            return (
                "Repair requirement: replace `solve satisfy:` with `solve satisfy;`. "
                "MiniZinc solve statements terminate with a semicolon, not a colon."
            )
        if "bare assignment" in lower_message:
            return (
                "Repair requirement: replace every bare assignment after declarations "
                "with a `constraint ...;` statement, or use an initialized parameter "
                "declaration such as `int: april_clips = 48;`."
            )
        if "output" in lower_message:
            return (
                "Repair requirement: rewrite the output item as an array of strings "
                "using string literals and `show(...)` for numeric values."
            )
    return ""
