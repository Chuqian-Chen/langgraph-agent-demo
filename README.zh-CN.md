# LangGraph Agent Demo

这个 demo 按照 LangGraph 官方 overview 页面完成：
https://docs.langchain.com/oss/python/langgraph/overview

目标不是做一个复杂聊天机器人，而是把 overview 里的最小可运行结构讲清楚：什么是图、状态如何流动、节点如何更新状态，以及 agent 每一步如何完成。

## 官方页面内容对应关系

1. 安装 LangGraph

官方页面先给出安装命令：

```powershell
pip install -U langgraph
```

本项目把依赖写在 `requirements.txt`：

```text
langgraph
```

在当前环境里，我把依赖安装到了项目内 `.deps` 目录，避免影响全局 Anaconda：

```powershell
python -m pip install -r requirements.txt --target .deps
```

2. Hello World 图

官方 overview 的最小思路是：

```text
START -> node -> END
```

本项目实现为：

```text
START -> mock_llm -> END
```

3. 核心概念

LangGraph 的 agent 不是一段单向脚本，而是一个状态图：

- `StateGraph`：定义图结构。
- `MessagesState`：保存消息列表，是这个 agent 的状态。
- `START`：图入口。
- `mock_llm`：业务节点，模拟一次 LLM 回复。
- `END`：图结束。
- `compile()`：把图编译成可以 `.invoke()` 的 runnable。

4. LangGraph 的能力边界

overview 还强调 LangGraph 适合构建长期运行、可控、可观测的 agent。这个 demo 没有接真实模型和生产部署，但结构上对应这些能力：

- 持久执行：真实项目可在 graph 里接 checkpoint，让执行中断后恢复。
- 人类介入：真实项目可在关键节点前暂停，让人审阅或批准。
- 记忆：当前 demo 用 `MessagesState` 保存消息；真实项目可加入长期记忆或数据库。
- 调试与观测：真实项目可接 LangSmith 查看每个节点的输入输出。
- 部署：真实项目可用 LangGraph Platform 部署 graph。

## 我设计的 agent

这个 agent 由两个公开函数组成：

```python
build_agent_graph()
run_demo(user_text: str)
```

设计原因：

- `build_agent_graph()` 只负责构建 LangGraph 图，便于测试图是否可调用。
- `run_demo()` 负责准备输入状态、调用图、整理输出，便于 CLI 和测试复用。
- `mock_llm()` 是图里的唯一节点，用固定回复模拟真实 LLM，避免 API key 和网络模型调用影响学习。

## 每一步如何执行

### 第 1 步：用户输入进入初始状态

输入文本：

```text
你好，LangGraph
```

被包装成：

```python
{"messages": [HumanMessage(content="你好，LangGraph")]}
```

这就是 graph 的初始状态。

### 第 2 步：从 START 进入 mock_llm 节点

代码：

```python
builder.add_edge(START, "mock_llm")
```

含义：

```text
图开始后，第一站是 mock_llm。
```

### 第 3 步：mock_llm 读取状态并返回状态更新

代码：

```python
def mock_llm(state: MessagesState) -> dict[str, list[AIMessage]]:
    return {"messages": [AIMessage(content="hello world")]}
```

它没有直接修改原状态，而是返回一个状态更新：

```python
{"messages": [AIMessage(content="hello world")]}
```

`MessagesState` 会把新消息追加到消息列表里。

### 第 4 步：从 mock_llm 到 END

代码：

```python
builder.add_edge("mock_llm", END)
```

含义：

```text
mock_llm 执行完后，图结束。
```

### 第 5 步：拿到最终状态

最终消息状态包含两条消息：

```text
HumanMessage: 你好，LangGraph
AIMessage: hello world
```

`run_demo()` 返回：

```python
{
    "reply": "hello world",
    "trace": ["START", "mock_llm", "END"],
    "messages": [...]
}
```

## 文件结构

```text
langgraph-agent-demo/
  langgraph_agent_demo/
    __init__.py
    demo.py
  tests/
    test_demo.py
  run_demo.py
  requirements.txt
  README.zh-CN.md
```

## 运行方式

进入项目目录：

```powershell
cd E:\test_for_codex\langgraph-agent-demo
```

安装依赖。常规 Python 环境可以这样：

```powershell
python -m pip install -r requirements.txt
```

如果你想像本次操作一样把依赖放在项目目录：

```powershell
python -m pip install -r requirements.txt --target .deps
```

运行测试：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest discover -s tests -v
```

运行 demo：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python run_demo.py
```

预期输出：

```text
LangGraph Agent Demo
========================
用户输入: 你好，LangGraph
执行路径:
  1. START
  2. mock_llm
  3. END
消息状态:
  - HumanMessage: 你好，LangGraph
  - AIMessage: hello world
最终回复: hello world
```

## 这次我是如何完成的

1. 读取官方 overview 页面，确认页面的主体是安装、Hello World graph、核心能力和生态说明。
2. 检查当前工作区，发现只有一个静态网页项目，所以新建独立目录 `langgraph-agent-demo`。
3. 先写测试，验证期望接口：
   - `run_demo()` 返回 `hello world`。
   - 执行路径是 `START -> mock_llm -> END`。
   - `build_agent_graph()` 返回可 `.invoke()` 的 graph。
4. 第一次运行测试，确认失败原因是包不存在。
5. 写最小实现：
   - `MessagesState` 作为状态。
   - `mock_llm` 作为唯一节点。
   - `START` 连到 `mock_llm`，再连到 `END`。
6. 安装 `langgraph` 依赖到 `.deps`。
7. 重新运行测试，确认 2 个测试通过。
8. 增加 `run_demo.py`，用命令行打印输入、执行路径、消息状态和最终回复。
