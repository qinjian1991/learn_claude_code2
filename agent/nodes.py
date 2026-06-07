from itertools import count
import logging

from core.config import settings
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.runtime import Runtime

from agent.logging_utils import log_messages_snapshot
from agent.memory import load_long_term_memories
from agent.memory_consolidation import consolidate_long_term_memories
from agent.memory_extraction import extract_long_term_memories
from agent.model import BASE_MODEL, MODEL
from agent.prompt import build_system_prompt
from agent.state import AgentState, RuntimeContext, state_snapshot
from agent.tools import SPECIAL_PANORAMA_SUMMARY_TOOL, TOOLS, TOOLS_BY_NAME
from micro_compact import micro_compact_messages
from panorama_summary import (
    create_panorama_summary_messages,
    estimate_messages_tokens,
    summarize_if_over_token_threshold,
)
from reactive_compact import is_context_error, reactive_compact_messages
from snip_compact import snip_compact
from tool_result_budget import tool_result_budget


logger = logging.getLogger("agent")
MODEL_EXCHANGE_COUNTER = count(1)


def prepare_model_input(state: AgentState, runtime: Runtime[RuntimeContext]) -> dict:
    long_term_memories = load_long_term_memories(runtime, state["messages"])
    system_prompt = build_system_prompt(runtime.context, long_term_memories)
    state_meta = state_snapshot(state)

    compacted_messages = tool_result_budget(state["messages"])
    compacted_messages = snip_compact(compacted_messages)
    compacted_messages = micro_compact_messages(compacted_messages)

    system_message = SystemMessage(content=system_prompt)

    def token_counter(messages):
        return BASE_MODEL.get_num_tokens_from_messages(
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

    logger.info("Calling model with %s conversation messages", len(state["messages"]))

    return {
        "model_input_messages": messages,
        "compacted_messages": compacted_messages,
        "last_token_estimate": estimated_tokens,
        "last_context_action": last_context_action,
        "summary_count": summary_count,
        "reactive_compact_count": reactive_compact_count,
        "last_compacted_turn": last_compacted_turn,
    }


def invoke_model(state: AgentState) -> dict:
    state_meta = state_snapshot(state)
    messages = state["model_input_messages"]
    compacted_messages = state["compacted_messages"]
    system_message = messages[0]
    reactive_compact_count = state_meta["reactive_compact_count"]
    last_context_action = state_meta["last_context_action"]
    last_compacted_turn = state_meta["last_compacted_turn"]

    reactive_retry_count = max(settings.reactive_compact_retry_count, 0)
    max_attempts = reactive_retry_count + 1
    for attempt in range(max_attempts):
        exchange_id = next(MODEL_EXCHANGE_COUNTER)
        label = "request messages"
        if attempt:
            label = f"reactive retry #{attempt} request messages"
        log_messages_snapshot(f"Model exchange #{exchange_id} {label}", messages)

        try:
            response = MODEL.invoke(messages)
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

    return {
        "model_response": response,
        "model_exchange_id": exchange_id,
        "model_input_messages": messages,
        "compacted_messages": compacted_messages,
        "last_context_action": last_context_action,
        "reactive_compact_count": reactive_compact_count,
        "last_compacted_turn": last_compacted_turn,
    }


def process_model_response(state: AgentState) -> dict:
    state_meta = state_snapshot(state)
    response = state["model_response"]
    compacted_messages = state["compacted_messages"]
    exchange_id = state["model_exchange_id"]

    tool_calls = getattr(response, "tool_calls", None) or []
    logger.info("Model response received; tool calls: %s", len(tool_calls))

    log_messages_snapshot(f"Model exchange #{exchange_id} response message", [response])

    return {
        "messages": compacted_messages + [response],
        "model_call_count": state_meta["model_call_count"] + 1,
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
        reason = str(tool_args.get("reason") or "Model requested context compaction.")
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
    logger.info("Routing to agent loop exit")
    return "exit_agent_loop"


def exit_agent_loop(state: AgentState, runtime: Runtime[RuntimeContext]) -> dict:
    state_meta = state_snapshot(state)
    logger.info(
        "Agent loop exiting; turn=%s model_calls=%s messages=%s",
        state_meta["conversation_turn"],
        state_meta["model_call_count"],
        len(state.get("messages") or []),
    )
    touched_categories = extract_long_term_memories(state, runtime)
    consolidate_long_term_memories(runtime, touched_categories)
    return {}
