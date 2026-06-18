from __future__ import annotations

from collections import Counter
from typing import Any


TODO_STATUSES = ("pending", "in_progress", "completed", "cancelled")


def normalize_todos(raw_todos: Any) -> list[dict[str, str]]:
    if raw_todos is None:
        return []
    if not isinstance(raw_todos, list):
        raise ValueError("todos must be a list.")

    normalized = []
    for index, item in enumerate(raw_todos, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"todo #{index} must be an object.")

        content = _normalize_content(item.get("content"))
        if not content:
            raise ValueError(f"todo #{index} is missing content.")

        status = str(item.get("status") or "").strip().lower()
        if status not in TODO_STATUSES:
            raise ValueError(
                f"todo #{index} has invalid status {status!r}; "
                f"expected one of {', '.join(TODO_STATUSES)}."
            )

        normalized.append({"content": content, "status": status})

    return normalized


def todo_update_summary(todos: list[dict[str, str]]) -> str:
    counts = Counter(todo["status"] for todo in todos)
    parts = [
        f"{status}={counts.get(status, 0)}"
        for status in TODO_STATUSES
        if counts.get(status, 0)
    ]
    if not parts:
        return "Updated CURRENT_TODOS: 0 items."
    return f"Updated CURRENT_TODOS: {len(todos)} items ({', '.join(parts)})."


def render_current_todos(todos: list[dict] | None) -> str:
    normalized = normalize_todos(todos or [])
    lines = ["CURRENT_TODOS:"]

    if normalized:
        for index, todo in enumerate(normalized, start=1):
            lines.append(f"{index}. [{todo['status']}] {todo['content']}")
    else:
        lines.append("(none)")

    lines.extend(
        [
            "",
            "TODO planning reminders:",
            "- For non-trivial multi-step work, call run_todo_write before acting.",
            "- Keep exactly the current execution focus marked in_progress.",
            "- Mark completed work promptly.",
            "- Update TODOs when the plan changes.",
            "- Before the final response, complete, cancel, or clear TODOs.",
        ]
    )
    return "\n".join(lines)


def _normalize_content(value: Any) -> str:
    return " ".join(str(value or "").split())
