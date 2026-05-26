import sys

from langgraph_agent_demo import run_demo


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    configure_stdout()
    user_text = "你好，LangGraph"
    result = run_demo(user_text)

    print("LangGraph Agent Demo")
    print("=" * 24)
    print(f"用户输入: {user_text}")
    print("执行路径:")
    for index, step in enumerate(result["trace"], start=1):
        print(f"  {index}. {step}")
    print("消息状态:")
    for message in result["messages"]:
        print(f"  - {message.__class__.__name__}: {message.content}")
    print(f"最终回复: {result['reply']}")


if __name__ == "__main__":
    main()
