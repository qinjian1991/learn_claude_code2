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
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Write UTF-8 text to a workspace file."""
    file_path = resolve_workspace_path(path)
    logger.info("Writing file: %s", file_path)

    if file_path.exists() and not overwrite:
        return f"File already exists: {path}. Set overwrite=true to replace it."

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {relative_workspace_path(file_path)}"


register_permission_rules([
    {
        "tools": ["write_file"],
        "action": "deny",
        "check": lambda args: file_path_missing(args),
        "message": "Missing file path.",
        "risk_summary": lambda args: path_risk_summary(args),
    },
    {
        "tools": ["write_file"],
        "action": "deny",
        "check": lambda args: file_path_outside_workspace(args),
        "message": "Path is outside the workspace.",
        "risk_summary": lambda args: path_risk_summary(args),
    },
    {
        "tools": ["write_file"],
        "action": "deny",
        "check": lambda args: file_path_targets_git(args),
        "message": "Modifying .git is denied.",
        "risk_summary": lambda args: path_risk_summary(args),
    },
    {
        "tools": ["write_file"],
        "action": "approval_required",
        "check": lambda args: True,
        "message": "File modification requires approval.",
        "risk_summary": lambda args: tool_args_risk_summary(args),
    },
])
