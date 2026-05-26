from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph


class DemoResult(TypedDict):
    reply: str
    trace: list[str]
    messages: list[object]


def mock_llm(state: MessagesState) -> dict[str, list[AIMessage]]:
    return {"messages": [AIMessage(content="hello world")]}


def build_agent_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("mock_llm", mock_llm)
    builder.add_edge(START, "mock_llm")
    builder.add_edge("mock_llm", END)
    return builder.compile()


def run_demo(user_text: str) -> DemoResult:
    graph = build_agent_graph()
    initial_state = {"messages": [HumanMessage(content=user_text)]}
    final_state = graph.invoke(initial_state)
    messages = final_state["messages"]
    return {
        "reply": messages[-1].content,
        "trace": ["START", "mock_llm", "END"],
        "messages": messages,
    }
