from __future__ import annotations

from typing import Any, Callable, Literal, TypedDict


PermissionAction = Literal["allow", "deny", "approval_required", "rejected"]


class PermissionRule(TypedDict):
    tools: list[str]
    action: PermissionAction
    check: Callable[[dict[str, Any]], bool]
    message: str
    risk_summary: Callable[[dict[str, Any]], str]


_PERMISSION_RULES: list[PermissionRule] = []


def register_permission_rules(rules: list[PermissionRule]) -> None:
    _PERMISSION_RULES.extend(rules)


def get_permission_rules() -> list[PermissionRule]:
    return list(_PERMISSION_RULES)
