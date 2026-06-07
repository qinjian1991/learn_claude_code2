from typing import Literal

from agent.state import RuntimeContext


MemoryCategory = Literal[
    "user_preferences",
    "user_feedback",
    "project_facts",
    "project_decisions",
    "project_references",
    "project_open_threads",
]


def memory_categories() -> tuple[MemoryCategory, ...]:
    return (
        "user_preferences",
        "user_feedback",
        "project_facts",
        "project_decisions",
        "project_references",
        "project_open_threads",
    )


def namespace_for_category(
    category: MemoryCategory,
    runtime_context: RuntimeContext,
) -> tuple[str, str, str]:
    if category == "user_preferences":
        return ("users", runtime_context["user_id"], "preferences")
    if category == "user_feedback":
        return ("users", runtime_context["user_id"], "feedback")
    if category == "project_facts":
        return ("projects", runtime_context["project_id"], "facts")
    if category == "project_decisions":
        return ("projects", runtime_context["project_id"], "decisions")
    if category == "project_references":
        return ("projects", runtime_context["project_id"], "references")
    return ("projects", runtime_context["project_id"], "open_threads")
