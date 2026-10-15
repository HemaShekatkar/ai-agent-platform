from app.tools.base import BaseTool
from app.tools.exceptions import (
    DuplicateToolError,
    ToolNotFoundError,
)


class ToolRegistry:

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        name = tool.definition.name

        if name in self._tools:
            raise DuplicateToolError(
                f"Tool '{name}' is already registered."
            )

        self._tools[name] = tool

    def get(self, name: str) -> BaseTool:
        tool = self._tools.get(name)

        if tool is None:
            raise ToolNotFoundError(
                f"Tool '{name}' is not registered."
            )

        return tool

    def definitions(self) -> list:
        return [
            tool.definition
            for tool in self._tools.values()
        ]
