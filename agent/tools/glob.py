import logging
from pathlib import Path

from langchain_core.tools import tool

from agent.tools.path_utils import relative_workspace_path
from core.constants import WORKDIR


logger = logging.getLogger("agent")


@tool("glob")
def glob_files(pattern: str) -> str:
    """Find workspace files matching a glob pattern."""
    logger.info("Globbing files with pattern: %s", pattern)

    if Path(pattern).is_absolute():
        return "Glob pattern must be relative to the workspace."

    workspace_path = WORKDIR.resolve()
    matches = [
        relative_workspace_path(path.resolve(strict=False))
        for path in workspace_path.glob(pattern)
        if path.is_file()
    ]
    matches.sort()

    if not matches:
        return "(no matches)"

    max_matches = 1000
    visible_matches = matches[:max_matches]
    output = "\n".join(visible_matches)
    if len(matches) > max_matches:
        output += f"\n... {len(matches) - max_matches} more matches"

    return output
