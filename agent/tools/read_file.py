import logging

from langchain_core.tools import tool

from agent.tools.path_utils import resolve_workspace_path
logger = logging.getLogger("agent")


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file from the workspace."""
    file_path = resolve_workspace_path(path)
    logger.info("Reading file: %s", file_path)

    if not file_path.exists():
        return f"File not found: {path}"
    if not file_path.is_file():
        return f"Path is not a file: {path}"

    return file_path.read_text(encoding="utf-8")
