from collections.abc import Iterator
from typing import Annotated

from core.config import settings
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


TOOLS = [add]


MODEL = ChatAnthropic(
    base_url=settings.anthropic_base_url,
    api_key=settings.anthropic_api_key,
    model_name=settings.anthropic_model,
    streaming=True,
    temperature=settings.model_temperature,
).bind_tools(TOOLS)


def call_model(state: AgentState) -> dict:
    response = MODEL.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")
    return builder.compile()


GRAPH = build_graph()


class Agent:
    def __init__(self) -> None:
        self.messages: list[BaseMessage] = []

    def stream(self, message: str) -> Iterator[str]:
        self.messages.append(HumanMessage(content=message))
        with GRAPH.stream_events(
            {"messages": self.messages},
            version="v3",
        ) as run:
            for message_stream in run.messages:
                if message_stream.node != "call_model":
                    continue
                yield from message_stream.text

            output = run.output
            if output:
                self.messages = output["messages"]

    def reset(self) -> None:
        self.messages = []


if __name__ == "__main__":
    if not settings.anthropic_api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY before running this example.")

    agent = Agent()
    for token in agent.stream("What is 12 + 30? Use the tool."):
        print(token, end="", flush=True)
    print()
