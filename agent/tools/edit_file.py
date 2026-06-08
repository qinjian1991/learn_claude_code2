import logging

from langchain_core.tools import tool

from agent.tools.path_utils import (
    file_path_missing,
    file_path_outside_workspace,
    file_path_targets_git,
    path_risk_summary,
    relative_workspace_path,
    resolve_workspace_path,
    tool_args_risk_summary,
)
from agent.tools.permission_registry import register_permission_rules


logger = logging.getLogger("agent")


@tool
def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> str:
    """Replace text in a UTF-8 workspace file."""
    file_path = resolve_workspace_path(path)
    logger.info("Editing file: %s", file_path)

    if not file_path.exists():
        return f"File not found: {path}"
    if not file_path.is_file():
        return f"Path is not a file: {path}"
    if old_text == "":
        return "old_text must not be empty."

    content = file_path.read_text(encoding="utf-8")
    match_count = content.count(old_text)

    if match_count == 0:
        return f"Text not found in {path}."
    if match_count > 1 and not replace_all:
        return (
            f"Found {match_count} matches in {path}. "
            "Set replace_all=true or provide a more specific old_text."
        )

    count = -1 if replace_all else 1
    updated_content = content.replace(old_text, new_text, count)
    file_path.write_text(updated_content, encoding="utf-8")
    replaced_count = match_count if replace_all else 1
    return f"Replaced {replaced_count} occurrence(s) in {relative_workspace_path(file_path)}"


register_permission_rules([
    {
        "tools": ["edit_file"],
        "action": "deny",
        "check": lambda args: file_path_missing(args),
        "message": "Missing file path.",
        "risk_summary": lambda args: path_risk_summary(args),
    },
    {
        "tools": ["edit_file"],
        "action": "deny",
        "check": lambda args: file_path_outside_workspace(args),
        "message": "Path is outside the workspace.",
        "risk_summary": lambda args: path_risk_summary(args),
    },
    {
        "tools": ["edit_file"],
        "action": "deny",
        "check": lambda args: file_path_targets_git(args),
        "message": "Modifying .git is denied.",
        "risk_summary": lambda args: path_risk_summary(args),
    },
    {
        "tools": ["edit_file"],
        "action": "approval_required",
        "check": lambda args: True,
        "message": "File modification requires approval.",
        "risk_summary": lambda args: tool_args_risk_summary(args),
    },
])
