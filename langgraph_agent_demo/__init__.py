"""对外公开 LangGraph multi-agent demo 的稳定入口。

外部代码只需要从包根导入 `build_agent_graph` 和 `run_demo`。
内部模块可以继续拆分，但这里保持最小公开 API。
"""

from .demo import run_demo
from .graph import build_agent_graph

__all__ = ["build_agent_graph", "run_demo"]
