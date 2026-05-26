from collections.abc import Callable


Tool = Callable[[str], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, tool: Tool) -> None:
        self._tools[name] = tool

    def run(self, name: str, query: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Tool not found: {name}"
        try:
            return tool(query)
        except Exception as exc:
            return f"Tool error from {name}: {exc}"


def build_default_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("research", lambda query: f"Mock research finding for {query}")
    return registry
