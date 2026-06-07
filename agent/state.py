from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    messages: list[BaseMessage]
    model_input_messages: list[BaseMessage]
    compacted_messages: list[BaseMessage]
    model_response: BaseMessage
    model_exchange_id: int | None
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
