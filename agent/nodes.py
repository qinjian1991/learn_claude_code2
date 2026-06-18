from itertools import count
import logging
import time

from core.config import settings
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from agent.logging_utils import log_messages_snapshot
from agent.memory import load_long_term_memories
from agent.memory_consolidation import consolidate_long_term_memories
from agent.memory_extraction import extract_long_term_memories
from agent.model import BASE_MODEL, MODEL
from agent.prompt import build_system_prompt
from agent.state import AgentState, RuntimeContext, state_snapshot
from agent.todos import normalize_todos, todo_update_summary
from agent.tools.permissions import (
    apply_approval_response,
    approval_payload,
    decision_by_tool_call_id,
    evaluate_tool_permissions,
)
from agent.tools import SPECIAL_PANORAMA_SUMMARY_TOOL, TOOLS, TOOLS_BY_NAME
from agent.tools.todo_write import run_todo_write
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
TODO_WRITE_TOOL = run_todo_write.name
MODEL_EXCHANGE_COUNTER = count(1)
MAX_TOKENS_ERROR_KEYWORDS = (
    "max_tokens",
    "max token",
    "maximum tokens",
    "maximum output",
    "output token",
    "output limit",
)
TRANSIENT_MODEL_ERROR_KEYWORDS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "overloaded",
    "temporarily unavailable",
    "service unavailable",
    "internal server error",
    "bad gateway",
    "gateway timeout",
    "timeout",
    "timed out",
    "connection",
    "network",
    "api connection",
)
TRANSIENT_MODEL_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def is_max_tokens_error(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    return any(keyword in text for keyword in MAX_TOKENS_ERROR_KEYWORDS)


def exception_status_code(exc: Exception) -> int | None:
    for value in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return None


def is_transient_model_error(exc: Exception) -> bool:
    status_code = exception_status_code(exc)
    if status_code in TRANSIENT_MODEL_STATUS_CODES:
        return True

    text = f"{exc.__class__.__name__}: {exc}".lower()
    if any(f" {code} " in f" {text} " for code in TRANSIENT_MODEL_STATUS_CODES):
        return True

    return any(keyword in text for keyword in TRANSIENT_MODEL_ERROR_KEYWORDS)


def retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(exc, "headers", None)
    if headers is None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None

    try:
        return max(float(retry_after), 0)
    except (TypeError, ValueError):
        return None


def transient_retry_delay(exc: Exception, retry_index: int) -> float:
    retry_after = retry_after_seconds(exc)
    if retry_after is not None:
        return min(retry_after, settings.model_transient_retry_max_seconds)

    initial_delay = max(settings.model_transient_retry_initial_seconds, 0)
    delay = initial_delay * (2 ** retry_index)
    return min(delay, settings.model_transient_retry_max_seconds)


def invoke_model_with_transient_retries(messages, exchange_id: int):
    retry_count = max(settings.model_transient_retry_count, 0)
    max_attempts = retry_count + 1

    for transient_attempt in range(max_attempts):
        try:
            return MODEL.invoke(messages)
        except Exception as exc:
            if not is_transient_model_error(exc) or transient_attempt >= retry_count:
                if is_transient_model_error(exc):
                    logger.exception(
                        "Model transient error persisted after %s retries: %s",
                        retry_count,
                        exc,
                    )
                    raise RuntimeError(
                        "Model call failed after retrying a transient provider error "
                        f"({exception_status_code(exc) or exc.__class__.__name__})."
                    ) from exc
                raise

            delay = transient_retry_delay(exc, transient_attempt)
            logger.warning(
                "Model exchange #%s transient error; retrying %s/%s in %.1fs: %s",
                exchange_id,
                transient_attempt + 1,
                retry_count,
                delay,
                exc,
            )
            if delay > 0:
                time.sleep(delay)


def model_stop_reason(response) -> str | None:
    response_metadata = getattr(response, "response_metadata", None) or {}
    stop_reason = response_metadata.get("stop_reason")
    if stop_reason:
        return str(stop_reason)

    additional_kwargs = getattr(response, "additional_kwargs", None) or {}
    stop_reason = additional_kwargs.get("stop_reason")
    if stop_reason:
        return str(stop_reason)

    return None


def prepare_model_input(state: AgentState, runtime: Runtime[RuntimeContext]) -> dict:
    long_term_memories = load_long_term_memories(runtime, state["messages"])
    state_meta = state_snapshot(state)
    system_prompt = build_system_prompt(
        runtime.context,
        long_term_memories,
        current_todos=state_meta["current_todos"],
    )

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
            response = invoke_model_with_transient_retries(messages, exchange_id)
            break
        except Exception as exc:
            if not is_context_error(exc) or attempt >= reactive_retry_count:
                if is_max_tokens_error(exc):
                    logger.exception(
                        "Model call failed with a max_tokens-related error. "
                        "Current MODEL_MAX_TOKENS=%s.",
                        settings.model_max_tokens,
                    )
                    raise RuntimeError(
                        "Model call failed with a max_tokens-related error. "
                        "Try lowering MODEL_MAX_TOKENS if the provider rejects the "
                        "request, or raising it if the response is being cut short. "
                        f"Current value: {settings.model_max_tokens}."
                    ) from exc

                logger.exception("Model call failed without a recoverable context error")
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
    if model_stop_reason(response) == "max_tokens":
        logger.warning(
            "Model response stopped because max_tokens was reached. "
            "Current MODEL_MAX_TOKENS=%s; response may be incomplete.",
            settings.model_max_tokens,
        )

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
    decisions = decision_by_tool_call_id(state.get("tool_permission_decisions") or [])
    todo_update = None
    todo_update_count = state_meta["todo_update_count"]

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
        decision = decisions.get(tool_call_id)
        action = (decision or {}).get("action")

        if action in {"deny", "rejected"}:
            reason = (decision or {}).get("reason") or "Tool call was not allowed."
            content = f"Tool call {action}: {reason}"
            tool_messages.append(
                ToolMessage(
                    content=content,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            )
            continue

        if action != "allow":
            content = "Tool call denied: no permission decision was available."
            tool_messages.append(
                ToolMessage(
                    content=content,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            )
            continue

        if tool_name == TODO_WRITE_TOOL:
            try:
                todos = normalize_todos(tool_args.get("todos"))
            except ValueError as exc:
                content = f"Tool error: {exc}"
            else:
                todo_update = todos
                todo_update_count += 1
                content = todo_update_summary(todos)

            tool_messages.append(
                ToolMessage(
                    content=content,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            )
            continue

        selected_tool = TOOLS_BY_NAME.get(tool_name)
        if selected_tool is None:
            content = f"Tool error: unknown tool {tool_name}"
            tool_messages.append(
                ToolMessage(
                    content=content,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            )
            continue

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
    result = {
        "messages": state["messages"] + tool_messages,
        "tool_permission_decisions": [],
        "tool_permission_reasons": {},
    }
    if todo_update is not None:
        result.update(
            {
                "current_todos": todo_update,
                "todo_update_count": todo_update_count,
                "last_todo_update_turn": state_meta["conversation_turn"],
            }
        )
    return result


def check_tool_permissions(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    decisions = evaluate_tool_permissions(tool_calls)
    approval_required = [
        decision
        for decision in decisions
        if decision["action"] == "approval_required"
    ]

    if approval_required:
        payload = approval_payload(decisions)
        logger.info(
            "Tool approval required; tool calls: %s",
            len(payload["tool_calls"]),
        )
        approval_response = interrupt(payload)
        decisions = apply_approval_response(decisions, approval_response)

    reasons = {
        decision["tool_call_id"]: decision["reason"]
        for decision in decisions
    }

    denied_count = sum(
        1
        for decision in decisions
        if decision["action"] in {"deny", "rejected"}
    )
    logger.info(
        "Tool permission check finished; decisions=%s denied_or_rejected=%s",
        len(decisions),
        denied_count,
    )
    return {
        "tool_permission_decisions": decisions,
        "tool_permission_reasons": reasons,
    }


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        logger.info("Routing to tool permission check")
        return "check_tool_permissions"
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
