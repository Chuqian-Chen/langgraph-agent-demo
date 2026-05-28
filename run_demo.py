import sys

from langgraph_agent_demo import run_demo


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    configure_stdout()
    user_text = "你好，LangGraph"
    result = run_demo(user_text)
    state = result["state"]

    print("LangGraph Multi-Agent Demo")
    print("=" * 28)
    print(f"用户输入: {user_text}")
    print("执行路径:")
    for index, step in enumerate(result["trace"], start=1):
        print(f"  {index}. {step}")
    print("共享状态:")
    print(f"  - plan: {state['plan']}")
    print(f"  - findings: {state['findings']}")
    print(f"  - status: {state['status']}")
    print(f"  - review: {state['review']}")
    print(f"最终回复: {result['reply']}")
    evolution = result["evolution"]
    evaluation = evolution["evaluation"]
    reflection = evolution["reflection"]
    proposal = evolution["proposal"]
    print("受控自进化:")
    print(f"  - evaluation score: {evaluation['score']}")
    print(f"  - evaluation passed: {evaluation['passed']}")
    print(f"  - strengths: {reflection['strengths']}")
    print(f"  - weaknesses: {reflection['weaknesses']}")
    print(f"  - proposal title: {proposal['title']}")
    print(f"  - proposal target: {proposal['target']}")
    print(f"  - proposal change: {proposal['change']}")
    print(f"  - requires_human_approval: {proposal['requires_human_approval']}")
    print(f"  - applied: {proposal['applied']}")


if __name__ == "__main__":
    main()
