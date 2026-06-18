from langchain_core.tools import tool

from agent.task_store import claim_task, complete_task, create_task, task_graph_json
from core.config import settings


def _owner() -> str:
    return f"agent:{settings.project_id}"


@tool
def task_list(compact: bool = True) -> str:
    """List the persistent cross-session task DAG."""
    return task_graph_json(compact=compact)


@tool
def task_create(
    title: str,
    description: str,
    depends_on: list[str] | None = None,
) -> str:
    """Create a pending task in the persistent DAG."""
    try:
        task = create_task(title, description, depends_on or [])
    except ValueError as exc:
        return f"Task error: {exc}"

    return f"Created task {task['id']}: {task['title']}"


@tool
def task_claim(task_id: str) -> str:
    """Claim a ready pending task whose dependencies are completed."""
    try:
        task = claim_task(task_id, _owner())
    except ValueError as exc:
        return f"Task error: {exc}"

    return f"Claimed task {task['id']}: {task['title']}"


@tool
def task_complete(task_id: str, result: str = "") -> str:
    """Complete a claimed task and store a concise result summary."""
    try:
        task = complete_task(task_id, result)
    except ValueError as exc:
        return f"Task error: {exc}"

    return f"Completed task {task['id']}: {task['title']}"
