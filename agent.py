from collections.abc import Iterator
from itertools import count
import json
import logging
import sqlite3
import subprocess

from core.config import settings
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.sqlite import SqliteStore
from typing_extensions import TypedDict

from core.constants import WORKDIR
from micro_compact import micro_compact_messages
from panorama_summary import (
    create_panorama_summary_messages,
    estimate_messages_tokens,
    summarize_if_over_token_threshold,
)
from reactive_compact import is_context_error, reactive_compact_messages
from snip_compact import serialize_message, snip_compact
from thread_memory import get_or_create_default_thread_id, rotate_default_thread_id
from tool_result_budget import tool_result_budget


logger = logging.getLogger("agent")
context_logger = logging.getLogger("agent.context")
MODEL_EXCHANGE_COUNTER = count(1)
CHECKPOINT_DB_PATH = WORKDIR / settings.checkpoint_db_path
CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
CHECKPOINT_CONNECTION = sqlite3.connect(
    CHECKPOINT_DB_PATH,
    check_same_thread=False,
)
CHECKPOINTER = SqliteSaver(CHECKPOINT_CONNECTION)
CHECKPOINTER.setup()
STORE_DB_PATH = WORKDIR / settings.store_db_path
STORE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
STORE_CONNECTION = sqlite3.connect(
    STORE_DB_PATH,
    check_same_thread=False,
    isolation_level=None,
)
STORE = SqliteStore(STORE_CONNECTION)
STORE.setup()


class AgentState(TypedDict, total=False):
    messages: list[BaseMessage]
    conversation_turn: int
    model_call_count: int
    last_token_estimate: int | None
    last_context_action: str | None
    summary_count: int
    reactive_compact_count: int
    model_requested_summary_count: int
    last_compacted_turn: int | None


class RuntimeContext(TypedDict):
    workspace: str
    user_id: str
    project_id: str


def state_counter(state: AgentState | dict, field: str) -> int:
    return int(state.get(field) or 0)


def state_snapshot(state: AgentState | dict) -> dict:
    return {
        "conversation_turn": state_counter(state, "conversation_turn"),
        "model_call_count": state_counter(state, "model_call_count"),
        "last_token_estimate": state.get("last_token_estimate"),
        "last_context_action": state.get("last_context_action") or "none",
        "summary_count": state_counter(state, "summary_count"),
        "reactive_compact_count": state_counter(state, "reactive_compact_count"),
        "model_requested_summary_count": state_counter(
            state,
            "model_requested_summary_count",
        ),
        "last_compacted_turn": state.get("last_compacted_turn"),
    }


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
    logger.info("PowerShell command finished with exit code %s",
                result.returncode)

    if result.returncode != 0:
        return f"Exit code: {result.returncode}\nSTDOUT:\n{output}\nSTDERR:\n{error}"

    return output or "(no output)"


@tool
def request_panorama_summary(reason: str) -> str:
    """Request a full conversation summary when the context is getting noisy or long."""
    return "Panorama summary requested."


SPECIAL_PANORAMA_SUMMARY_TOOL = request_panorama_summary.name
TOOLS = [powershell, request_panorama_summary]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


BASE_MODEL = ChatAnthropic(
    base_url=settings.anthropic_base_url,
    api_key=settings.anthropic_api_key,
    model_name=settings.anthropic_model,
    streaming=True,
    temperature=settings.model_temperature,
)
MODEL = BASE_MODEL.bind_tools(TOOLS)


PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "context_compaction": (
        "If the conversation context becomes noisy, repetitive, or hard to reason about, "
        "call request_panorama_summary to compact the full conversation before continuing."
    ),
}


def build_system_prompt(
    runtime_context: RuntimeContext,
    long_term_memories: list[str] | None = None,
) -> str:
    sections = []

    sections.append(PROMPT_SECTIONS["identity"])
    sections.append(f"Working directory: {runtime_context['workspace']}")
    if long_term_memories:
        sections.append("Long-term memory:\n" + "\n".join(long_term_memories))
    sections.append(PROMPT_SECTIONS["context_compaction"])

    return "\n\n".join(sections)


def call_model(state: AgentState, runtime: Runtime[RuntimeContext]) -> dict:

    long_term_memories = load_long_term_memories(runtime)
    system_prompt = build_system_prompt(runtime.context, long_term_memories)
    state_meta = state_snapshot(state)

    compacted_messages = tool_result_budget(state["messages"])
    compacted_messages = snip_compact(compacted_messages)
    compacted_messages = micro_compact_messages(compacted_messages)

    system_message = SystemMessage(content=system_prompt)

    def token_counter(messages): return BASE_MODEL.get_num_tokens_from_messages(
        messages,
        tools=TOOLS,
    )
    estimated_tokens = estimate_messages_tokens(
        [system_message] + compacted_messages,
        token_counter,
    )
    last_context_action = "message_compact"
    summary_count = state_meta["summary_count"]
    reactive_compact_count = state_meta["reactive_compact_count"]
    last_compacted_turn = state_meta["last_compacted_turn"]
    if estimated_tokens > settings.context_token_threshold:
        last_context_action = "token_threshold_summary"
        summary_count += 1
        last_compacted_turn = state_meta["conversation_turn"]

    compacted_messages = summarize_if_over_token_threshold(
        compacted_messages,
        token_counter=token_counter,
        summarizer=BASE_MODEL,
        threshold=settings.context_token_threshold,
        prefix_messages=[system_message],
    )

    messages = [system_message] + compacted_messages

    logger.info("Calling model with %s conversation messages",
                len(state["messages"]))

    reactive_retry_count = max(settings.reactive_compact_retry_count, 0)
    max_attempts = reactive_retry_count + 1
    for attempt in range(max_attempts):
        exchange_id = next(MODEL_EXCHANGE_COUNTER)
        label = "request messages"
        if attempt:
            label = f"reactive retry #{attempt} request messages"
        log_messages_snapshot(
            f"Model exchange #{exchange_id} {label}", messages)

        try:
            response = MODEL.invoke(
                messages
            )
            break
        except Exception as exc:
            if not is_context_error(exc) or attempt >= reactive_retry_count:
                raise

            logger.info(
                "Model context error detected; applying reactive compact retry %s/%s: %s",
                attempt + 1,
                reactive_retry_count,
                exc,
            )
            compacted_messages = reactive_compact_messages(
                compacted_messages,
                BASE_MODEL,
                reason=f"Main model failed with context error: {exc}",
            )
            reactive_compact_count += 1
            last_context_action = "reactive_compact"
            last_compacted_turn = state_meta["conversation_turn"]
            messages = [system_message] + compacted_messages

    tool_calls = getattr(response, "tool_calls", None) or []
    logger.info("Model response received; tool calls: %s", len(tool_calls))

    log_messages_snapshot(
        f"Model exchange #{exchange_id} response message", [response])

    return {
        "messages": compacted_messages + [response],
        "conversation_turn": state_meta["conversation_turn"],
        "model_call_count": state_meta["model_call_count"] + 1,
        "last_token_estimate": estimated_tokens,
        "last_context_action": last_context_action,
        "summary_count": summary_count,
        "reactive_compact_count": reactive_compact_count,
        "model_requested_summary_count": state_meta["model_requested_summary_count"],
        "last_compacted_turn": last_compacted_turn,
    }


def call_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []
    state_meta = state_snapshot(state)

    for tool_call in getattr(last_message, "tool_calls", None) or []:
        tool_name = tool_call["name"]
        if tool_name != SPECIAL_PANORAMA_SUMMARY_TOOL:
            continue

        tool_args = tool_call.get("args") or {}
        reason = str(tool_args.get("reason")
                     or "Model requested context compaction.")
        logger.info("Panorama summary tool requested: %s", reason)
        summarized_messages = create_panorama_summary_messages(
            state["messages"],
            BASE_MODEL,
            reason=reason,
        )
        logger.info(
            "Panorama summary tool completed; messages: %s -> %s",
            len(state["messages"]),
            len(summarized_messages),
        )
        return {
            "messages": summarized_messages,
            "conversation_turn": state_meta["conversation_turn"],
            "model_call_count": state_meta["model_call_count"],
            "last_token_estimate": state_meta["last_token_estimate"],
            "last_context_action": "model_requested_summary",
            "summary_count": state_meta["summary_count"] + 1,
            "reactive_compact_count": state_meta["reactive_compact_count"],
            "model_requested_summary_count": (
                state_meta["model_requested_summary_count"] + 1
            ),
            "last_compacted_turn": state_meta["conversation_turn"],
        }

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


def load_long_term_memories(runtime: Runtime[RuntimeContext]) -> list[str]:
    if runtime.store is None:
        return []

    user_id = runtime.context["user_id"]
    project_id = runtime.context["project_id"]
    memory_items = []
    for namespace in (
        ("users", user_id, "preferences"),
        ("projects", project_id, "facts"),
    ):
        memory_items.extend(runtime.store.search(namespace, limit=10))

    memories = [
        format_memory_item(item)
        for item in memory_items
    ]
    if memories:
        logger.info("Loaded long-term memories: %s", len(memories))
    return memories


def format_memory_item(item) -> str:
    namespace = "/".join(item.namespace)
    return f"- [{namespace}/{item.key}] {json.dumps(item.value, ensure_ascii=False)}"


def build_graph():
    builder = StateGraph(AgentState, context_schema=RuntimeContext)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", call_tools)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")
    return builder.compile(
        checkpointer=CHECKPOINTER,
        store=STORE,
    )


GRAPH = build_graph()


class Agent:
    def __init__(self, thread_id: str | None = None) -> None:
        self.thread_id = thread_id or get_or_create_default_thread_id()
        logger.info("Agent initialized; thread_id: %s", self.thread_id)

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    @property
    def runtime_context(self) -> RuntimeContext:
        return {
            "workspace": str(WORKDIR),
            "user_id": settings.user_id,
            "project_id": settings.project_id,
        }

    def get_messages(self) -> list[BaseMessage]:
        return list(self.get_state_values().get("messages") or [])

    def get_state_values(self) -> dict:
        state = GRAPH.get_state(self.config)
        return dict(state.values if state else {})

    def stream(self, message: str) -> Iterator[str]:
        state_values = self.get_state_values()
        state_meta = state_snapshot(state_values)
        messages = list(state_values.get("messages") or [])
        messages.append(HumanMessage(content=message))
        logger.info("User message received; message length: %s", len(message))
        with GRAPH.stream_events(
            {
                **state_meta,
                "messages": messages,
                "conversation_turn": state_meta["conversation_turn"] + 1,
                "last_context_action": "user_message",
            },
            self.config,
            context=self.runtime_context,
            version="v3",
        ) as run:
            for message_stream in run.messages:
                if message_stream.node != "call_model":
                    continue
                yield from message_stream.text

            output = run.output
            if output:
                logger.info("Agent stream completed; stored messages: %s", len(
                    output["messages"]))

    def reset(self) -> None:
        CHECKPOINTER.delete_thread(self.thread_id)
        logger.info("Agent reset; thread_id: %s", self.thread_id)

    def new_thread(self) -> str:
        self.thread_id = rotate_default_thread_id()
        logger.info("Agent switched to new thread; thread_id: %s",
                    self.thread_id)
        return self.thread_id

    def remember_user_preference(self, key: str, value: dict) -> None:
        STORE.put(("users", settings.user_id, "preferences"), key, value)
        logger.info("Stored user preference memory: %s", key)

    def remember_project_fact(self, key: str, value: dict) -> None:
        STORE.put(("projects", settings.project_id, "facts"), key, value)
        logger.info("Stored project fact memory: %s", key)

    def get_long_term_memories(self) -> list[dict]:
        items = []
        for namespace in (
            ("users", settings.user_id, "preferences"),
            ("projects", settings.project_id, "facts"),
        ):
            items.extend(STORE.search(namespace, limit=50))

        return [
            {
                "namespace": item.namespace,
                "key": item.key,
                "value": item.value,
            }
            for item in items
        ]


if __name__ == "__main__":
    if not settings.anthropic_api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY before running this example.")

    agent = Agent()
    for token in agent.stream("What is the current directory? Use the PowerShell tool."):
        print(token, end="", flush=True)
    print()
