from pathlib import Path

from core.constants import WORKDIR


def resolve_workspace_path(path: str) -> Path:
    requested_path = Path(path)
    if not requested_path.is_absolute():
        requested_path = WORKDIR / requested_path

    resolved_path = requested_path.resolve(strict=False)
    workspace_path = WORKDIR.resolve()

    try:
        resolved_path.relative_to(workspace_path)
    except ValueError as exc:
        raise ValueError(f"Path is outside workspace: {path}") from exc

    return resolved_path


def relative_workspace_path(path: Path) -> str:
    return path.relative_to(WORKDIR.resolve()).as_posix()


def resolve_relative_workspace_path(path_value) -> Path | None:
    path = str(path_value or "")
    if not path:
        return None

    requested_path = Path(path)
    if not requested_path.is_absolute():
        requested_path = WORKDIR / requested_path

    resolved_path = requested_path.resolve(strict=False)
    workspace_path = WORKDIR.resolve()
    try:
        return resolved_path.relative_to(workspace_path)
    except ValueError:
        return None


def file_path_missing(args: dict) -> bool:
    return not str(args.get("path") or "")


def file_path_outside_workspace(args: dict) -> bool:
    return resolve_relative_workspace_path(args.get("path")) is None


def file_path_targets_git(args: dict) -> bool:
    relative_path = resolve_relative_workspace_path(args.get("path"))
    return relative_path is not None and ".git" in relative_path.parts


def path_risk_summary(args: dict) -> str:
    return str(args.get("path") or "")


def tool_args_risk_summary(args: dict) -> str:
    return str(args)
