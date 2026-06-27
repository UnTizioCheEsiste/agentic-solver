import pytest

from agentic_solver.mcp_server.backends.base import SolveResult, ValidationResult
from agentic_solver.mcp_server.server import create_mcp_server
from agentic_solver.mcp_server.session import ModelSession


class FakeBackend:
    name = "minizinc"

    def validate(self, model: str) -> ValidationResult:
        return ValidationResult(valid=True, status="valid", message=f"valid: {model}")

    def solve(self, model: str) -> SolveResult:
        return SolveResult(status="sat", solution=f"solution: {model}", message="solved")


@pytest.mark.anyio
async def test_mcp_server_registers_fixed_tool_interface() -> None:
    server = create_mcp_server(ModelSession(backends={"minizinc": FakeBackend()}))

    tools = await server.list_tools()
    tool_names = {tool.name for tool in tools}

    assert tool_names >= {
        "clear_model",
        "add_item",
        "replace_item",
        "delete_item",
        "get_model",
        "validate_model",
        "solve_model",
    }

