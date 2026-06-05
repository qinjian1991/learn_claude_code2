import logging

from langchain_core.messages import BaseMessage

from panorama_summary import create_panorama_summary_messages


KEEP_RECENT_ROUNDS = 5

logger = logging.getLogger("agent")


CONTEXT_ERROR_KEYWORDS = (
    "context length",
    "context window",
    "context_window_exceeded",
    "maximum context",
    "max context",
    "prompt is too long",
    "input is too long",
    "input tokens",
    "too many tokens",
    "token limit",
    "exceeds",
    "request too large",
    "request_too_large",
)


def is_context_error(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    return any(keyword in text for keyword in CONTEXT_ERROR_KEYWORDS)


def reactive_compact_messages(
    messages: list[BaseMessage],
    summarizer,
    *,
    keep_recent_rounds: int = KEEP_RECENT_ROUNDS,
    reason: str | None = None,
) -> list[BaseMessage]:
    split_index = find_recent_round_split_index(messages, keep_recent_rounds)
    older_messages = messages[:split_index]
    recent_messages = messages[split_index:]

    if not older_messages:
        logger.info(
            "Reactive compact skipped; no older messages before last %s rounds",
            keep_recent_rounds,
        )
        return messages

    summary_reason = reason or (
        "Main model returned a context-related error; compact older history and "
        f"keep the last {keep_recent_rounds} user rounds verbatim."
    )
    summarized_older_messages = create_panorama_summary_messages(
        older_messages,
        summarizer,
        reason=summary_reason,
    )
    compacted_messages = summarized_older_messages + recent_messages
    logger.info(
        "Reactive compact completed; older messages summarized: %s, recent messages kept: %s, total: %s -> %s",
        len(older_messages),
        len(recent_messages),
        len(messages),
        len(compacted_messages),
    )
    return compacted_messages


def find_recent_round_split_index(
    messages: list[BaseMessage],
    keep_recent_rounds: int,
) -> int:
    if keep_recent_rounds <= 0:
        return len(messages)

    rounds_seen = 0
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].type != "human":
            continue

        rounds_seen += 1
        if rounds_seen == keep_recent_rounds:
            return index

    return 0
