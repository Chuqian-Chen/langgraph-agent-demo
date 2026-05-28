"""工具注册层。

Agent 节点通过 `ToolRegistry` 查找工具，避免直接依赖具体工具实现。
后续接搜索、文件读取或代码执行工具时，只需要扩展注册表。
"""

from collections.abc import Callable


Tool = Callable[[str], str]


class ToolRegistry:
    """按名称注册和调用字符串输入/输出工具。"""

    def __init__(self) -> None:
        """创建一个空工具注册表。"""

        self._tools: dict[str, Tool] = {}

    def register(self, name: str, tool: Tool) -> None:
        """注册工具函数，后注册的同名工具会覆盖旧工具。"""

        self._tools[name] = tool

    def run(self, name: str, query: str) -> str:
        """运行指定工具，并把缺失工具或工具异常转换成可观察文本。"""

        tool = self._tools.get(name)
        if tool is None:
            return f"Tool not found: {name}"
        try:
            return tool(query)
        except Exception as exc:
            return f"Tool error from {name}: {exc}"


def build_default_tools() -> ToolRegistry:
    """构建 demo 默认工具集，目前只包含 mock research 工具。"""

    registry = ToolRegistry()
    registry.register("research", lambda query: f"Mock research finding for {query}")
    return registry
