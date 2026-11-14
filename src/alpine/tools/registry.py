import logging
from pathlib import Path
from typing import Any

from alpine.tools.base_tool import Tool, ToolInvocation, ToolResult

logger = logging.getLogger(__name__)

class ToolRegistry:
    def __init__(self) -> None:
        self._tools = dict[str, Tool] = {}

    def registry(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")

            self._tools[tool.name] = tool
            logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True

        return False

    def get(self, name: str) -> Tool | None:
        if name in self._tools:
            return self._tools[name]

        return None

    def get_tools(self) -> list[Tool]:
        tools: list[Tool] = []

        for tool in self._tools.values():
            tools.append(tool)

        return tools

    def get_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self.get_tools()]

    async def invoke(self, name: str, params: dict[str, Any], cwd: Path | None):
        tool = self.get(name)
        if tool is None:
            return ToolResult.err_result(f"Tool does not exists: {name}", metadata={"tool_name": name})

        validation_errs = tool.validate_params(params)
        if validation_errs:
            return ToolResult.err_result(
                    f"Invalid parameters: {'; '.join(validation_errs)}",
                    metadata={
                        'tool_name': name,
                        "validation_errors": validation_errs,
                        },
                    )

        invocations = ToolInvocation(
                params,
                cwd,
                ) 
        try:
            await tool.execute(invocations)
        except Exception as e:
            pass
