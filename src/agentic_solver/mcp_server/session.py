"""Mutable model-building session behind the MCP tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentic_solver.mcp_server.backends import AspBackend, MiniZincBackend, PrologBackend
from agentic_solver.mcp_server.backends.base import (
    SolveResult,
    SolverBackend,
    SolverName,
    ValidationResult,
)


class ModelItem(BaseModel):
    """A replaceable piece of the current solver model."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)


class ToolResponse(BaseModel):
    """Stable JSON-compatible response returned by MCP tools."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class ModelSession:
    """Stateful session for one model construction attempt."""

    def __init__(self, backends: Mapping[SolverName, SolverBackend] | None = None) -> None:
        self.backends: dict[SolverName, SolverBackend] = dict(backends or _default_backends())
        self.selected_solver: SolverName | None = None
        self.items: list[ModelItem] = []
        self.tool_calls = 0
        self.repair_attempts = 0
        self._last_validation_valid: bool | None = None

    def clear_model(self, solver: SolverName) -> ToolResponse:
        self._count_tool_call()
        if solver not in self.backends:
            return ToolResponse(ok=False, message=f"Unsupported solver: {solver}")

        self.selected_solver = solver
        self.items.clear()
        self._last_validation_valid = None
        return ToolResponse(ok=True, message="Model cleared.", data=self._state_payload())

    def add_item(self, content: str) -> ToolResponse:
        self._count_tool_call()
        if not content.strip():
            return ToolResponse(ok=False, message="Item content cannot be empty.")

        self.items.append(ModelItem(content=content))
        return ToolResponse(
            ok=True,
            message="Item added.",
            data=self._state_payload(index=len(self.items)),
        )

    def replace_item(self, index: int, content: str) -> ToolResponse:
        self._count_tool_call()
        if not content.strip():
            return ToolResponse(ok=False, message="Replacement content cannot be empty.")

        zero_based_index = self._zero_based_index(index)
        if zero_based_index is None:
            return ToolResponse(ok=False, message=f"Unknown item index: {index}")

        self.items[zero_based_index] = ModelItem(content=content)
        self._count_repair_if_needed()
        return ToolResponse(ok=True, message="Item replaced.", data=self._state_payload(index=index))

    def delete_item(self, index: int) -> ToolResponse:
        self._count_tool_call()
        zero_based_index = self._zero_based_index(index)
        if zero_based_index is None:
            return ToolResponse(ok=False, message=f"Unknown item index: {index}")

        del self.items[zero_based_index]
        self._count_repair_if_needed()
        return ToolResponse(
            ok=True,
            message="Item deleted. Remaining items have been reindexed.",
            data=self._state_payload(deleted_index=index),
        )

    def get_model(self) -> ToolResponse:
        self._count_tool_call()
        return ToolResponse(ok=True, message="Current model returned.", data=self._state_payload())

    def validate_model(self) -> ToolResponse:
        self._count_tool_call()
        backend_response = self._backend()
        if isinstance(backend_response, ToolResponse):
            return backend_response

        result = backend_response.validate(self.model_text)
        self._last_validation_valid = result.valid
        return ToolResponse(
            ok=result.valid,
            message=result.message,
            data={
                **self._state_payload(),
                "validation": result.model_dump(),
            },
        )

    def solve_model(self) -> ToolResponse:
        self._count_tool_call()
        backend_response = self._backend()
        if isinstance(backend_response, ToolResponse):
            return backend_response

        result = backend_response.solve(self.model_text)
        return ToolResponse(
            ok=result.status == "sat",
            message=result.message,
            data={
                **self._state_payload(),
                "solve": result.model_dump(),
            },
        )

    @property
    def model_text(self) -> str:
        return "\n".join(item.content for item in self.items)

    def _backend(self) -> SolverBackend | ToolResponse:
        if self.selected_solver is None:
            return ToolResponse(ok=False, message="No solver selected. Call clear_model first.")
        return self.backends[self.selected_solver]

    def _count_tool_call(self) -> None:
        self.tool_calls += 1

    def _count_repair_if_needed(self) -> None:
        if self._last_validation_valid is False:
            self.repair_attempts += 1
            self._last_validation_valid = None

    def _zero_based_index(self, index: int) -> int | None:
        if index < 1 or index > len(self.items):
            return None
        return index - 1

    def _state_payload(self, **extra: Any) -> dict[str, Any]:
        return {
            "selected_solver": self.selected_solver,
            "tool_calls": self.tool_calls,
            "repair_attempts": self.repair_attempts,
            "items": [
                {"index": index, "content": item.content}
                for index, item in enumerate(self.items, start=1)
            ],
            "model": self.model_text,
            **extra,
        }


def _default_backends() -> dict[SolverName, SolverBackend]:
    return {
        "asp": AspBackend(),
        "prolog": PrologBackend(),
        "minizinc": MiniZincBackend(),
    }
