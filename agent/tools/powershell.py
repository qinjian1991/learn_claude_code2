import logging
import re
import subprocess

from langchain_core.tools import tool

from core.constants import WORKDIR
from agent.tools.permission_registry import register_permission_rules


logger = logging.getLogger("agent")


@tool
def powershell(command: str) -> str:
    """Run a PowerShell command and return its output."""
    logger.info("Running PowerShell command: %s", command)
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        cwd=WORKDIR,
        text=True,
        timeout=30,
    )

    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    logger.info("PowerShell command finished with exit code %s", result.returncode)

    if result.returncode != 0:
        return f"Exit code: {result.returncode}\nSTDOUT:\n{output}\nSTDERR:\n{error}"

    return output or "(no output)"


DENIED_PATTERNS = [
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\b.*-[a-z]*[fdx][a-z]*",
    r"\b(remove-item|rm|del|erase|rmdir|rd)\b.*(\.git\b|\\\.git\b|/\.git\b)",
    r"\b(remove-item|rm|del|erase|rmdir|rd)\b.*\s(-recurse|-r)\b.*\s(\.|\./|\.\\)\s*$",
]
APPROVAL_PATTERNS = [
    r"\b(remove-item|rm|del|erase|rmdir|rd)\b",
    r"\b(move-item|mv|ren|rename-item)\b",
    r"\b(set-content|add-content|out-file|new-item|copy-item)\b",
    r"\b(git\s+(push|commit|merge|rebase|checkout|cherry-pick|pull|fetch))\b",
    r"\b(pip|python\s+-m\s+pip|npm|pnpm|yarn)\s+(install|add|remove|uninstall)\b",
    r"\b(curl|wget|iwr|irm|invoke-webrequest|invoke-restmethod)\b",
]


def _command_empty(args: dict) -> bool:
    return not str(args.get("command") or "").strip()


def _command_matches(args: dict, patterns: list[str]) -> bool:
    command = _normalize_command(str(args.get("command") or ""))
    return any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in patterns)


def _normalize_command(command: str) -> str:
    return " ".join(command.lower().split())


def _risk_summary(args: dict) -> str:
    return str(args.get("command") or "").strip()


register_permission_rules([
    {
        "tools": ["powershell"],
        "action": "deny",
        "check": lambda args: _command_empty(args),
        "message": "Empty command.",
        "risk_summary": lambda args: _risk_summary(args),
    },
    {
        "tools": ["powershell"],
        "action": "deny",
        "check": lambda args: _command_matches(args, DENIED_PATTERNS),
        "message": "Command is in the deny list.",
        "risk_summary": lambda args: _risk_summary(args),
    },
    {
        "tools": ["powershell"],
        "action": "approval_required",
        "check": lambda args: _command_matches(args, APPROVAL_PATTERNS),
        "message": "Potentially mutating or network command requires approval.",
        "risk_summary": lambda args: _risk_summary(args),
    },
])
