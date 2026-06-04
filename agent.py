from collections.abc import Iterator
import json
import logging
import subprocess

from core.config import settings
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from core.constants import WORKDIR
from core.log_config import configure_logging
from micro_compact import micro_compact_messages
from snip_compact import serialize_message, snip_compact
from tool_result_budget import tool_result_budget


configure_logging()
logger = logging.getLogger("agent")
context_logger = logging.getLogger("agent.context")


class AgentState(TypedDict):
    messages: list[BaseMessage]


def log_messages_snapshot(label: str, messages: list[BaseMessage]) -> None:
    context_logger.debug(
        "%s\n%s",
        label,
        json.dumps(
            {
                "message_count": len(messages),
                "messages": [serialize_message(message) for message in messages],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
    )


@tool
def powershell(command: str) -> str:
    """Run a PowerShell command and return its output."""
    logger.info("Running PowerShell command: %s", command)
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        cwd=WORKDIR,
    )

    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    logger.info("PowerShell command finished with exit code %s", result.returncode)

    if result.returncode != 0:
        return f"Exit code: {result.returncode}\nSTDOUT:\n{output}\nSTDERR:\n{error}"

    return output or "(no output)"


TOOLS = [powershell]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


MODEL = ChatAnthropic(
    base_url=settings.anthropic_base_url,
    api_key=settings.anthropic_api_key,
    model_name=settings.anthropic_model,
    streaming=True,
    temperature=settings.model_temperature,
).bind_tools(TOOLS)


PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "workspace": f"Working directory: {WORKDIR}",
}

def build_system_prompt() -> str:
    sections = []

    sections.append(PROMPT_SECTIONS["identity"])
    sections.append(PROMPT_SECTIONS["workspace"])

    return "\n\n".join(sections)

def call_model(state: AgentState) -> dict:

    system_prompt = build_system_prompt()

    compacted_messages = tool_result_budget(state["messages"])
    compacted_messages = snip_compact(compacted_messages)
    compacted_messages = micro_compact_messages(compacted_messages)
    
    messages = [SystemMessage(content=system_prompt)] + compacted_messages

    logger.info("Calling model with %s conversation messages", len(state["messages"]))

    log_messages_snapshot("Before model invoke: persisted state messages", state["messages"])
    log_messages_snapshot("Before model invoke: actual messages sent to model", messages)

    response = MODEL.invoke(
        messages
    )

    tool_calls = getattr(response, "tool_calls", None) or []
    logger.info("Model response received; tool calls: %s", len(tool_calls))

    log_messages_snapshot("After model invoke: model response message", [response])
    log_messages_snapshot("After model invoke: projected persisted messages", compacted_messages + [response])

    return {"messages": compacted_messages + [response]}


def call_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in getattr(last_message, "tool_calls", None) or []:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args") or {}
        tool_call_id = tool_call["id"]
        selected_tool = TOOLS_BY_NAME[tool_name]

        try:
            content = selected_tool.invoke(tool_args)
        except Exception as exc:
            content = f"Tool error: {exc}"

        tool_messages.append(
            ToolMessage(
                content=str(content),
                name=tool_name,
                tool_call_id=tool_call_id,
            )
        )

    logger.info("Tool node finished; tool messages: %s", len(tool_messages))
    return {"messages": state["messages"] + tool_messages}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        logger.info("Routing to tools")
        return "tools"
    logger.info("Routing to end")
    return END


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", call_tools)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")
    return builder.compile()


GRAPH = build_graph()


class Agent:
    def __init__(self) -> None:
        self.messages: list[BaseMessage] = []
        logger.info("Agent initialized")

    def stream(self, message: str) -> Iterator[str]:
        self.messages.append(HumanMessage(content=message))
        logger.info("User message received; message length: %s", len(message))
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
                logger.info("Agent stream completed; stored messages: %s", len(self.messages))

    def reset(self) -> None:
        self.messages = []
        logger.info("Agent reset")


if __name__ == "__main__":
    if not settings.anthropic_api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY before running this example.")

    agent = Agent()
    for token in agent.stream("What is the current directory? Use the PowerShell tool."):
        print(token, end="", flush=True)
    print()
