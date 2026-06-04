import json
import logging

from langchain_core.messages import BaseMessage, HumanMessage


MAX_CONTEXT_MESSAGES = 50
KEEP_HEAD_MESSAGES = 3


context_logger = logging.getLogger("agent.context")


def serialize_message(message: BaseMessage) -> dict:
    data = {
        "type": message.type,
        "class": message.__class__.__name__,
        "content": message.content,
        "content_length": len(str(message.content)),
    }

    for field in ("id", "name", "additional_kwargs", "response_metadata", "tool_calls"):
        value = getattr(message, field, None)
        if value:
            data[field] = value

    return data


def get_tool_call_ids(message: BaseMessage) -> list[str]:
    tool_calls = getattr(message, "tool_calls", None) or []
    tool_call_ids = []

    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            tool_call_id = tool_call.get("id")
        else:
            tool_call_id = getattr(tool_call, "id", None)

        if tool_call_id:
            tool_call_ids.append(tool_call_id)

    return tool_call_ids


def get_tool_message_call_id(message: BaseMessage) -> str | None:
    return getattr(message, "tool_call_id", None)


def remove_incomplete_tool_pairs(messages: list[BaseMessage]) -> list[BaseMessage]:
    ai_tool_calls_by_index = {}
    tool_message_indexes_by_call_id = {}

    for index, message in enumerate(messages):
        tool_call_ids = get_tool_call_ids(message)
        if tool_call_ids:
            ai_tool_calls_by_index[index] = tool_call_ids

        tool_call_id = get_tool_message_call_id(message)
        if tool_call_id:
            tool_message_indexes_by_call_id.setdefault(tool_call_id, []).append(index)

    tool_call_owner_by_id = {
        tool_call_id: index
        for index, tool_call_ids in ai_tool_calls_by_index.items()
        for tool_call_id in tool_call_ids
    }

    indexes_to_remove = set()

    for index, tool_call_ids in ai_tool_calls_by_index.items():
        missing_tool_results = [
            tool_call_id
            for tool_call_id in tool_call_ids
            if tool_call_id not in tool_message_indexes_by_call_id
        ]
        if not missing_tool_results:
            continue

        indexes_to_remove.add(index)
        for tool_call_id in tool_call_ids:
            indexes_to_remove.update(tool_message_indexes_by_call_id.get(tool_call_id, []))

    for tool_call_id, indexes in tool_message_indexes_by_call_id.items():
        if tool_call_id not in tool_call_owner_by_id:
            indexes_to_remove.update(indexes)

    if indexes_to_remove:
        context_logger.debug(
            "Removed incomplete tool pairs\n%s",
            json.dumps(
                {
                    "removed_indexes": sorted(indexes_to_remove),
                    "removed_messages": [
                        serialize_message(messages[index])
                        for index in sorted(indexes_to_remove)
                    ],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
        )

    return [
        message
        for index, message in enumerate(messages)
        if index not in indexes_to_remove
    ]


def snip_compact(messages: list[BaseMessage]) -> list[BaseMessage]:
    if len(messages) <= MAX_CONTEXT_MESSAGES:
        return messages

    tail_count = MAX_CONTEXT_MESSAGES - KEEP_HEAD_MESSAGES - 1
    snipped_count = len(messages) - KEEP_HEAD_MESSAGES - tail_count
    snipped_message = HumanMessage(
        content=f"[snipped {snipped_count} messages from conversation middle]"
    )
    compacted = messages[:KEEP_HEAD_MESSAGES] + [snipped_message] + messages[-tail_count:]
    compacted = remove_incomplete_tool_pairs(compacted)

    context_logger.debug(
        "Snip compact applied\n%s",
        json.dumps(
            {
                "max_context_messages": MAX_CONTEXT_MESSAGES,
                "keep_head_messages": KEEP_HEAD_MESSAGES,
                "before_count": len(messages),
                "after_count": len(compacted),
                "snipped_middle_count": snipped_count,
                "removed_count": len(messages) - len(compacted),
                "kept_message_types": [message.type for message in compacted],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
    )

    return compacted
