import json
import logging

from langchain_core.messages import BaseMessage

from snip_compact import serialize_message


context_logger = logging.getLogger("agent.context")


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
