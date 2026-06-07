import logging
import subprocess

from core.constants import WORKDIR
from langchain_core.tools import tool


logger = logging.getLogger("agent")


@tool
def powershell(command: str) -> str:
    """Run a PowerShell command and return its output."""
    logger.info("Running PowerShell command: %s", command)
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        cwd=WORKDIR,
    )

    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    logger.info("PowerShell command finished with exit code %s", result.returncode)

    if result.returncode != 0:
        return f"Exit code: {result.returncode}\nSTDOUT:\n{output}\nSTDERR:\n{error}"

    return output or "(no output)"


@tool
def request_panorama_summary(reason: str) -> str:
    """Request a full conversation summary when the context is getting noisy or long."""
    return "Panorama summary requested."


SPECIAL_PANORAMA_SUMMARY_TOOL = request_panorama_summary.name
TOOLS = [powershell, request_panorama_summary]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
