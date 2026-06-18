from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

from core.config import settings
from core.constants import WORKDIR


TASK_STATUSES = ("pending", "claimed", "completed", "cancelled")
TERMINAL_TASK_STATUSES = ("completed", "cancelled")
TASK_GRAPH_VERSION = 1


def task_graph_path(path: str | Path | None = None) -> Path:
    requested_path = Path(path or settings.task_graph_path)
    if not requested_path.is_absolute():
        requested_path = WORKDIR / requested_path

    resolved_path = requested_path.resolve(strict=False)
    workspace_path = WORKDIR.resolve()
    try:
        resolved_path.relative_to(workspace_path)
    except ValueError as exc:
        raise ValueError(f"Task graph path is outside workspace: {path}") from exc

    return resolved_path


def empty_task_graph() -> dict[str, Any]:
    return {"version": TASK_GRAPH_VERSION, "tasks": []}


def load_task_graph(
    path: str | Path | None = None,
    *,
    initialize: bool = True,
) -> dict[str, Any]:
    graph_path = task_graph_path(path)
    if not graph_path.exists():
        graph = empty_task_graph()
        if initialize:
            save_task_graph(graph, graph_path)
        return graph

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    return normalize_task_graph(graph)


def save_task_graph(graph: dict[str, Any], path: str | Path | None = None) -> None:
    graph_path = task_graph_path(path)
    normalized = normalize_task_graph(graph)
    validate_task_graph(normalized)

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = graph_path.with_name(f"{graph_path.name}.tmp")
    temp_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, graph_path)


def create_task(
    title: str,
    description: str,
    depends_on: list[str] | None = None,
    *,
    path: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    graph = load_task_graph(path)
    tasks = graph["tasks"]
    existing_ids = {task["id"] for task in tasks}
    normalized_depends_on = _normalize_depends_on(depends_on)
    missing = [task_id for task_id in normalized_depends_on if task_id not in existing_ids]
    if missing:
        raise ValueError(f"Unknown dependency task id(s): {', '.join(missing)}")

    timestamp = _timestamp(now)
    task = {
        "id": _next_task_id(tasks, now),
        "title": _required_text(title, "title"),
        "description": _required_text(description, "description"),
        "depends_on": normalized_depends_on,
        "status": "pending",
        "owner": "",
        "created_at": timestamp,
        "updated_at": timestamp,
        "claimed_at": "",
        "completed_at": "",
        "result": "",
    }
    graph["tasks"].append(task)
    save_task_graph(graph, path)
    return task


def claim_task(
    task_id: str,
    owner: str,
    *,
    path: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    graph = load_task_graph(path)
    task = _find_task(graph, task_id)
    if task["status"] != "pending":
        raise ValueError(f"Task {task_id} cannot be claimed from status {task['status']}.")

    tasks_by_id = {item["id"]: item for item in graph["tasks"]}
    incomplete = [
        dependency_id
        for dependency_id in task["depends_on"]
        if tasks_by_id[dependency_id]["status"] != "completed"
    ]
    if incomplete:
        raise ValueError(
            f"Task {task_id} cannot be claimed until dependencies complete: "
            f"{', '.join(incomplete)}"
        )

    timestamp = _timestamp(now)
    task["status"] = "claimed"
    task["owner"] = _required_text(owner, "owner")
    task["claimed_at"] = timestamp
    task["updated_at"] = timestamp
    save_task_graph(graph, path)
    return task


def complete_task(
    task_id: str,
    result: str = "",
    *,
    path: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    graph = load_task_graph(path)
    task = _find_task(graph, task_id)
    if task["status"] != "claimed":
        raise ValueError(f"Task {task_id} cannot be completed from status {task['status']}.")

    timestamp = _timestamp(now)
    task["status"] = "completed"
    task["completed_at"] = timestamp
    task["updated_at"] = timestamp
    task["result"] = _normalize_text(result)
    save_task_graph(graph, path)
    return task


def task_graph_json(compact: bool = True, path: str | Path | None = None) -> str:
    graph = load_task_graph(path)
    value = compact_task_graph(graph) if compact else graph
    return json.dumps(value, ensure_ascii=False, indent=2)


def compact_task_graph(graph: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_task_graph(graph)
    tasks = normalized["tasks"]
    return {
        "version": normalized["version"],
        "task_count": len(tasks),
        "status_counts": dict(Counter(task["status"] for task in tasks)),
        "ready_to_claim": [
            _compact_task(task)
            for task in tasks
            if task["status"] == "pending" and _dependencies_complete(task, tasks)
        ],
        "claimed": [
            _compact_task(task)
            for task in tasks
            if task["status"] == "claimed"
        ],
        "tasks": [_compact_task(task) for task in tasks],
    }


def render_task_graph_summary(
    path: str | Path | None = None,
    *,
    max_items: int = 8,
) -> str:
    graph = load_task_graph(path, initialize=False)
    tasks = graph["tasks"]
    counts = Counter(task["status"] for task in tasks)
    lines = [
        "TASK_GRAPH:",
        f"Total tasks: {len(tasks)}",
        "Status counts: "
        + (
            ", ".join(
                f"{status}={counts.get(status, 0)}"
                for status in TASK_STATUSES
                if counts.get(status, 0)
            )
            or "none"
        ),
    ]

    ready = [
        task
        for task in tasks
        if task["status"] == "pending" and _dependencies_complete(task, tasks)
    ]
    claimed = [task for task in tasks if task["status"] == "claimed"]
    lines.append("Ready to claim:")
    lines.extend(_render_task_lines(ready, max_items))
    lines.append("Claimed:")
    lines.extend(_render_task_lines(claimed, max_items))

    if len(tasks) > max_items:
        lines.append("Task graph is truncated; call task_list for full details.")

    lines.extend(
        [
            "",
            "Task system reminders:",
            "- Use task_create to split durable cross-session work into a DAG.",
            "- Use task_claim before working a ready task.",
            "- Use task_complete with a concise result when a claimed task is done.",
            "- Use CURRENT_TODOS for short-term steps inside the active task.",
        ]
    )
    return "\n".join(lines)


def normalize_task_graph(graph: Any) -> dict[str, Any]:
    if not isinstance(graph, dict):
        raise ValueError("Task graph must be a JSON object.")

    tasks = graph.get("tasks") or []
    if not isinstance(tasks, list):
        raise ValueError("Task graph tasks must be a list.")

    normalized_tasks = [_normalize_task(task, index) for index, task in enumerate(tasks)]
    normalized = {
        "version": int(graph.get("version") or TASK_GRAPH_VERSION),
        "tasks": normalized_tasks,
    }
    validate_task_graph(normalized)
    return normalized


def validate_task_graph(graph: dict[str, Any]) -> None:
    tasks = graph.get("tasks") or []
    ids = [task["id"] for task in tasks]
    duplicate_ids = [task_id for task_id, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate task id(s): {', '.join(duplicate_ids)}")

    ids_set = set(ids)
    for task in tasks:
        missing = [
            dependency_id
            for dependency_id in task["depends_on"]
            if dependency_id not in ids_set
        ]
        if missing:
            raise ValueError(
                f"Task {task['id']} has unknown dependency id(s): {', '.join(missing)}"
            )

    _assert_acyclic(tasks)


def _normalize_task(task: Any, index: int) -> dict[str, Any]:
    if not isinstance(task, dict):
        raise ValueError(f"Task #{index + 1} must be an object.")

    status = str(task.get("status") or "pending").strip().lower()
    if status not in TASK_STATUSES:
        raise ValueError(f"Task {task.get('id') or index + 1} has invalid status {status!r}.")

    return {
        "id": _required_text(task.get("id"), "id"),
        "title": _required_text(task.get("title"), "title"),
        "description": _normalize_text(task.get("description")),
        "depends_on": _normalize_depends_on(task.get("depends_on")),
        "status": status,
        "owner": _normalize_text(task.get("owner")),
        "created_at": _normalize_text(task.get("created_at")),
        "updated_at": _normalize_text(task.get("updated_at")),
        "claimed_at": _normalize_text(task.get("claimed_at")),
        "completed_at": _normalize_text(task.get("completed_at")),
        "result": _normalize_text(task.get("result")),
    }


def _assert_acyclic(tasks: list[dict[str, Any]]) -> None:
    edges = {task["id"]: list(task["depends_on"]) for task in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            raise ValueError(f"Task graph contains a cycle involving {task_id}.")

        visiting.add(task_id)
        for dependency_id in edges.get(task_id, []):
            visit(dependency_id)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in edges:
        visit(task_id)


def _find_task(graph: dict[str, Any], task_id: str) -> dict[str, Any]:
    normalized_id = _required_text(task_id, "task_id")
    for task in graph["tasks"]:
        if task["id"] == normalized_id:
            return task
    raise ValueError(f"Unknown task id: {normalized_id}")


def _next_task_id(tasks: list[dict[str, Any]], now: datetime | None) -> str:
    date_part = (now or datetime.now()).strftime("%Y%m%d")
    prefix = f"task_{date_part}_"
    existing_numbers = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        if task_id.startswith(prefix):
            try:
                existing_numbers.append(int(task_id.removeprefix(prefix)))
            except ValueError:
                continue

    return f"{prefix}{max(existing_numbers, default=0) + 1:03d}"


def _timestamp(now: datetime | None) -> str:
    return (now or datetime.now()).isoformat(timespec="seconds")


def _required_text(value: Any, field: str) -> str:
    text = _normalize_text(value)
    if not text:
        raise ValueError(f"{field} is required.")
    return text


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _normalize_depends_on(depends_on: Any) -> list[str]:
    if depends_on is None:
        return []
    if not isinstance(depends_on, list):
        raise ValueError("depends_on must be a list.")

    normalized = []
    for item in depends_on:
        task_id = _required_text(item, "depends_on item")
        if task_id not in normalized:
            normalized.append(task_id)
    return normalized


def _dependencies_complete(task: dict[str, Any], tasks: list[dict[str, Any]]) -> bool:
    tasks_by_id = {item["id"]: item for item in tasks}
    return all(
        tasks_by_id[dependency_id]["status"] == "completed"
        for dependency_id in task["depends_on"]
    )


def _compact_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task["id"],
        "title": task["title"],
        "status": task["status"],
        "depends_on": task["depends_on"],
        "owner": task["owner"],
    }


def _render_task_lines(tasks: list[dict[str, Any]], max_items: int) -> list[str]:
    if not tasks:
        return ["- none"]
    return [
        f"- {task['id']} [{task['status']}] {task['title']}"
        for task in tasks[:max_items]
    ]
