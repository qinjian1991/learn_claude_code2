from datetime import datetime
from pathlib import Path

from langchain_core.messages import BaseMessage

from core.constants import WORKDIR


MAX_TOOL_RESULT_CHARS = 200_000
TOOL_RESULT_OUTPUT_DIR = WORKDIR / "tool_results"


def tool_result_budget(messages: list[BaseMessage]) -> list[BaseMessage]:
    TOOL_RESULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    budgeted_messages = []

    for index, message in enumerate(messages):
        if message.type != "tool":
            budgeted_messages.append(message)
            continue

        content = str(message.content)
        if len(content) <= MAX_TOOL_RESULT_CHARS:
            budgeted_messages.append(message)
            continue

        full_output_path = save_full_tool_result(message, content)
        budgeted_messages.append(
            compact_tool_result_message(message, content, full_output_path)
        )

    return budgeted_messages


def save_full_tool_result(message: BaseMessage, content: str) -> Path:
    tool_call_id = sanitize_filename(getattr(message, "tool_call_id", None) or "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = TOOL_RESULT_OUTPUT_DIR / f"{timestamp}_{tool_call_id}.txt"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def compact_tool_result_message(
    message: BaseMessage,
    content: str,
    full_output_path: Path,
) -> BaseMessage:
    compacted_content = compact_tool_result_content(content, full_output_path)

    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": compacted_content})
    if hasattr(message, "copy"):
        return message.copy(update={"content": compacted_content})

    return type(message)(
        content=compacted_content,
        name=getattr(message, "name", None),
        tool_call_id=getattr(message, "tool_call_id"),
    )


def compact_tool_result_content(content: str, full_output_path: Path) -> str:
    marker = (
        f"\n\n[snipped {len(content) - MAX_TOOL_RESULT_CHARS} characters from "
        "tool result middle; full output saved to "
        f"{full_output_path}]\n\n"
    )
    available_chars = max(MAX_TOOL_RESULT_CHARS - len(marker), 0)
    head_chars = available_chars // 2
    tail_chars = available_chars - head_chars

    return content[:head_chars] + marker + content[-tail_chars:]


def sanitize_filename(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in value
    )
