import logging

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from core.config import settings


logger = logging.getLogger("agent")


SUBAGENT_SYSTEM_PROMPT = """You are a focused subagent working on a delegated task.

Use tools when needed, but keep the work tightly scoped to the task. You have your
own independent message history. When you are done, answer with the final result
only; the parent agent will receive a concise summary, not your full transcript.
"""


def _coerce_max_turns(max_turns: int | None) -> int:
    default_turns = max(settings.subagent_default_max_turns, 1)
    requested_turns = default_turns if max_turns is None else int(max_turns)
    return min(max(requested_turns, 1), max(settings.subagent_max_turns, 1))


def _message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _subagent_tools():
    from agent.tools.registry import SPECIAL_PANORAMA_SUMMARY_TOOL, TOOLS

    blocked_tool_names = {
        spawn_subagent.name,
        SPECIAL_PANORAMA_SUMMARY_TOOL,
    }
    return [tool_item for tool_item in TOOLS if tool_item.name not in blocked_tool_names]


def _summarize_result(task: str, final_text: str, reached_limit: bool) -> str:
    from agent.model import BASE_MODEL

    limit_note = (
        "The subagent reached its turn limit before a natural stop."
        if reached_limit
        else "The subagent stopped naturally."
    )
    prompt = (
        "Summarize the subagent's final result for the parent agent. "
        "Return only the useful outcome, decisions, file paths, commands, errors, "
        "or next steps. Do not include the full transcript.\n\n"
        f"Task:\n{task}\n\n"
        f"Status:\n{limit_note}\n\n"
        f"Final result or last observation:\n{final_text}"
    )
    response = BASE_MODEL.invoke(
        [
            SystemMessage(content="You write concise handoff summaries."),
            HumanMessage(content=prompt),
        ]
    )
    summary = _message_text(response).strip()
    return summary or final_text


@tool
def spawn_subagent(task: str, max_turns: int | None = None) -> str:
    """Delegate a focused task to a subagent with its own messages and turn limit."""
    from agent.model import BASE_MODEL
    from agent.tools.permissions import evaluate_tool_permission
    from agent.tools.registry import TOOLS_BY_NAME

    turn_limit = _coerce_max_turns(max_turns)
    model = BASE_MODEL.bind_tools(_subagent_tools())
    messages = [
        SystemMessage(content=SUBAGENT_SYSTEM_PROMPT),
        HumanMessage(content=task),
    ]
    final_text = ""
    reached_limit = True

    logger.info("Spawning subagent with max_turns=%s", turn_limit)

    for turn_index in range(turn_limit):
        response = model.invoke(messages)
        messages.append(response)
        final_text = _message_text(response).strip()
        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            reached_limit = False
            break

        tool_messages = []
        for tool_call in tool_calls:
            tool_name = str(tool_call.get("name") or "")
            tool_args = tool_call.get("args") or {}
            tool_call_id = str(tool_call.get("id") or "")
            decision = evaluate_tool_permission(tool_call)

            if decision["action"] != "allow":
                content = (
                    "Tool call denied inside subagent: "
                    f"{decision['reason']}"
                )
            else:
                selected_tool = TOOLS_BY_NAME.get(tool_name)
                if selected_tool is None or selected_tool.name == spawn_subagent.name:
                    content = f"Tool error: unknown or unavailable tool {tool_name}"
                else:
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

        messages.extend(tool_messages)
        if tool_messages:
            final_text = "\n".join(_message_text(message) for message in tool_messages)
        logger.info(
            "Subagent turn %s/%s finished; tool calls=%s",
            turn_index + 1,
            turn_limit,
            len(tool_calls),
        )

    return _summarize_result(task, final_text, reached_limit)
