"""Build solver-specific models using the fixed MCP tool interface."""

from __future__ import annotations

import json
import sys
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_solver.agents.solver_selector import GenerateText, SolverName
from agentic_solver.config import DEFAULT_MODEL_ID
from agentic_solver.mcp_server.session import ModelSession


class ModelBuildPlan(BaseModel):
    """Initial model construction plan returned by the model-building agent."""

    model_config = ConfigDict(extra="forbid")

    items: list[str] = Field(min_length=1)
    output_contract: str = Field(min_length=1)


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
    max_repair_attempts: int = 2,
) -> dict[str, Any]:
    """Generate, validate, repair, and solve a model for the selected solver."""

    model_session = session or ModelSession()
    text_generator = generator or _build_transformers_generator(model_id)

    _log_progress("requesting initial solver model")
    initial_plan = parse_model_build_plan(text_generator(build_model_build_prompt(problem, solver)))

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
        if solve_payload["status"] != "error" or repair_rounds >= max_repair_attempts:
            break

        repair_rounds += 1
        _repair_model(
            model_session=model_session,
            text_generator=text_generator,
            problem=problem,
            solver=solver,
            model_state=solve_response.data,
            failure_stage="solve",
            failure_message=_summarize_failure(solve_response.message),
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
        "answer_artifact": {
            "problem": problem,
            "solver": solver,
            "model": model_session.model_text,
            "solver_status": solve_payload["status"] if solve_payload else "not_solved",
            "raw_solution": solve_payload["solution"] if solve_payload else None,
            "solver_message": solve_payload["message"] if solve_payload else validation_response.message,
            "output_contract": initial_plan.output_contract,
        },
    }


def build_model_build_prompt(problem: str, solver: SolverName) -> str:
    """Build the prompt for the model-building agent."""

    return (
        "You are the second agent in a deterministic symbolic-solving pipeline.\n"
        "Your task is to translate the natural-language problem into code for "
        f"the selected solver: {solver}.\n\n"
        "Return only valid JSON, without Markdown or extra text. The JSON must "
        "contain exactly these fields:\n"
        "- items: a non-empty array of solver-code strings. Each item should be "
        "a coherent replaceable block.\n"
        "- output_contract: a short description of how the raw solver output "
        "should be interpreted by a later natural-language answer agent.\n\n"
        "The solver code must be directly executable by the selected backend. "
        "Use explicit output directives when the solver supports them.\n\n"
        f"{_solver_specific_instructions(solver)}\n\n"
        f"Problem:\n{problem}"
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
            "- Recursive graph programs must terminate on cyclic graphs. Use a "
            "visited list or another cycle-safe strategy, and prefer setof/3 "
            "when duplicate answers should be removed.\n"
            "- For graph reachability from node 1, use this terminating pattern "
            "instead of direct recursive reachable/1 rules:\n"
            "  path(From, To, Visited) :- edge(From, To), \\+ member(To, Visited).\n"
            "  path(From, To, Visited) :- edge(From, Next), \\+ member(Next, Visited), "
            "path(Next, To, [Next|Visited]).\n"
            "  reachable(Node) :- path(1, Node, [1]).\n"
            "  solve(Result) :- setof(Node, reachable(Node), Result), !.\n"
            "  solve([])."
        )
    if solver == "asp":
        return (
            "ASP backend requirements:\n"
            "- Include #show directives for exactly the atoms that should be "
            "visible in the raw solution."
        )
    return (
        "MiniZinc backend requirements:\n"
        "- Include solve satisfy or an optimization objective.\n"
        "- Include an output item that prints the data needed by the answer agent."
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
        return (
            "Repair requirement: replace the recursive reachability code with a "
            "visited-list path/3 formulation. Do not keep direct rules like "
            "reachable(X) :- ..., reachable(Y), ... unless they include an explicit "
            "visited set that prevents revisiting nodes."
        )
    return ""
