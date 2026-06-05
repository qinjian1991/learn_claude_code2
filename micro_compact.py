from langchain_core.messages import BaseMessage


KEEP_LATEST_TOOL_BATCHES = 3
COMPACTED_TOOL_RESULT_CONTENT = "[Earlier tool result compacted. Re-run if needed.]"


def micro_compact_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    tool_batches = get_tool_batches(messages)
    if len(tool_batches) <= KEEP_LATEST_TOOL_BATCHES:
        return messages

    tool_call_ids_to_keep = {
        tool_call_id
        for batch in tool_batches[-KEEP_LATEST_TOOL_BATCHES:]
        for tool_call_id in batch["tool_call_ids"]
    }
    compacted_messages = []
    for index, message in enumerate(messages):
        if message.type != "tool":
            compacted_messages.append(message)
            continue

        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id in tool_call_ids_to_keep:
            compacted_messages.append(message)
            continue

        compacted_messages.append(compact_tool_message(message))

    return compacted_messages


def get_tool_batches(messages: list[BaseMessage]) -> list[dict]:
    batches = []

    for index, message in enumerate(messages):
        tool_calls = getattr(message, "tool_calls", None) or []
        tool_call_ids = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                tool_call_id = tool_call.get("id")
            else:
                tool_call_id = getattr(tool_call, "id", None)

            if tool_call_id:
                tool_call_ids.append(tool_call_id)

        if tool_call_ids:
            batches.append(
                {
                    "message_index": index,
                    "tool_call_ids": tool_call_ids,
                }
            )

    return batches


def compact_tool_message(message: BaseMessage) -> BaseMessage:
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": COMPACTED_TOOL_RESULT_CONTENT})
    if hasattr(message, "copy"):
        return message.copy(update={"content": COMPACTED_TOOL_RESULT_CONTENT})

    return type(message)(
        content=COMPACTED_TOOL_RESULT_CONTENT,
        name=getattr(message, "name", None),
        tool_call_id=getattr(message, "tool_call_id"),
    )
