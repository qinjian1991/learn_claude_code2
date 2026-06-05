import json
import logging
from collections.abc import Callable

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


PANORAMA_SUMMARY_TOKEN_THRESHOLD = 120_000
APPROX_CHARS_PER_TOKEN = 4

logger = logging.getLogger("agent")


SUMMARY_SYSTEM_PROMPT = """You are compacting a long coding-agent conversation.

Create a complete panorama summary that preserves everything needed to continue the work safely:
- current user goal and latest request
- important decisions, constraints, and assumptions
- files changed or inspected
- tool calls and their meaningful results
- unresolved tasks, risks, and next actions

Keep concrete filenames, commands, errors, and user preferences. Prefer concise structured prose. Do not invent facts."""


def summarize_if_over_token_threshold(
    messages: list[BaseMessage],
    *,
    token_counter: Callable[[list[BaseMessage]], int],
    summarizer,
    threshold: int = PANORAMA_SUMMARY_TOKEN_THRESHOLD,
    prefix_messages: list[BaseMessage] | None = None,
) -> list[BaseMessage]:
    prefix_messages = prefix_messages or []
    estimated_tokens = estimate_messages_tokens(prefix_messages + messages, token_counter)
    if estimated_tokens <= threshold:
        logger.info(
            "Context token estimate within threshold: %s <= %s",
            estimated_tokens,
            threshold,
        )
        return messages

    logger.info(
        "Context token estimate exceeded threshold; triggering panorama summary: %s > %s",
        estimated_tokens,
        threshold,
    )

    summarized_messages = create_panorama_summary_messages(messages, summarizer)
    summarized_tokens = estimate_messages_tokens(
        prefix_messages + summarized_messages,
        token_counter,
    )
    logger.info(
        "Panorama summary completed; messages: %s -> %s, token estimate: %s -> %s",
        len(messages),
        len(summarized_messages),
        estimated_tokens,
        summarized_tokens,
    )
    if summarized_tokens > threshold:
        logger.info(
            "Panorama summary still exceeds threshold: %s > %s",
            summarized_tokens,
            threshold,
        )
    return summarized_messages


def create_panorama_summary_messages(
    messages: list[BaseMessage],
    summarizer,
    *,
    reason: str | None = None,
) -> list[BaseMessage]:
    summary = create_panorama_summary(messages, summarizer, reason=reason)
    reason_text = f"\n\n[Summary reason]\n{reason}" if reason else ""
    return [
        HumanMessage(
            content=(
                "[Panorama summary of the conversation so far]"
                f"{reason_text}\n\n"
                f"{summary}\n\n"
                "[Continue from this summary as the complete prior conversation.]"
            )
        )
    ]


def estimate_messages_tokens(
    messages: list[BaseMessage],
    token_counter: Callable[[list[BaseMessage]], int],
) -> int:
    try:
        return token_counter(messages)
    except Exception as exc:
        logger.info("Model token counter failed; using approximate estimate: %s", exc)
        return approximate_messages_tokens(messages)


def approximate_messages_tokens(messages: list[BaseMessage]) -> int:
    rendered = render_messages_for_summary(messages)
    return max(len(rendered) // APPROX_CHARS_PER_TOKEN, 1)


def create_panorama_summary(
    messages: list[BaseMessage],
    summarizer,
    *,
    reason: str | None = None,
) -> str:
    transcript = render_messages_for_summary(messages)
    reason_text = f"\n\nReason for compacting now:\n{reason}" if reason else ""
    response = summarizer.invoke(
        [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=f"{transcript}{reason_text}"),
        ]
    )
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content)


def render_messages_for_summary(messages: list[BaseMessage]) -> str:
    return json.dumps(
        {
            "message_count": len(messages),
            "messages": [serialize_message_for_summary(message) for message in messages],
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def serialize_message_for_summary(message: BaseMessage) -> dict:
    data = {
        "type": message.type,
        "class": message.__class__.__name__,
        "content": message.content,
    }

    for field in ("id", "name", "additional_kwargs", "response_metadata", "tool_calls"):
        value = getattr(message, field, None)
        if value:
            data[field] = value

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        data["tool_call_id"] = tool_call_id

    return data
