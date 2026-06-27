"""FastMCP server exposing stable model-building tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from agentic_solver.mcp_server.backends.base import SolverName
from agentic_solver.mcp_server.session import ModelSession


def create_mcp_server(session: ModelSession | None = None) -> FastMCP:
    """Create the MCP server and register model-building tools."""

    model_session = session or ModelSession()
    server = FastMCP(
        "agentic-solver",
        instructions=(
            "Build, validate, repair, and solve symbolic solver models through "
            "the fixed clear/add/replace/delete/get/validate/solve tool interface."
        ),
    )

    @server.tool()
    def clear_model(solver: SolverName) -> dict:
        """Clear the current model and select asp, prolog, or minizinc."""

        return model_session.clear_model(solver).model_dump()

    @server.tool()
    def add_item(content: str) -> dict:
        """Add a replaceable item to the current model."""

        return model_session.add_item(content).model_dump()

    @server.tool()
    def replace_item(index: int, content: str) -> dict:
        """Replace an existing model item by its current 1-based index."""

        return model_session.replace_item(index, content).model_dump()

    @server.tool()
    def delete_item(index: int) -> dict:
        """Delete an existing model item by its current 1-based index."""

        return model_session.delete_item(index).model_dump()

    @server.tool()
    def get_model() -> dict:
        """Return the current model and session counters."""

        return model_session.get_model().model_dump()

    @server.tool()
    def validate_model() -> dict:
        """Validate the current model with the selected solver backend."""

        return model_session.validate_model().model_dump()

    @server.tool()
    def solve_model() -> dict:
        """Solve the current model with the selected solver backend."""

        return model_session.solve_model().model_dump()

    return server


def main() -> None:
    """Run the MCP server over stdio."""

    create_mcp_server().run()


if __name__ == "__main__":
    main()
