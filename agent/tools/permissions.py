from __future__ import annotations

from typing import Any, Literal, TypedDict

from agent.tools.permission_registry import PermissionRule, get_permission_rules
from agent.tools import registry as _tool_registry


PermissionAction = Literal["allow", "deny", "approval_required", "rejected"]


class ToolPermissionDecision(TypedDict):
    tool_call_id: str
    tool_name: str
    action: PermissionAction
    reason: str
    risk_summary: str


def permission_rules() -> list[PermissionRule]:
    _ = _tool_registry.TOOLS
    return get_permission_rules()


PERMISSION_RULES = permission_rules()


def refresh_permission_rules() -> list[PermissionRule]:
    global PERMISSION_RULES
    PERMISSION_RULES = permission_rules()
    return PERMISSION_RULES


def evaluate_tool_permissions(tool_calls: list[dict[str, Any]]) -> list[ToolPermissionDecision]:
    return [evaluate_tool_permission(tool_call) for tool_call in tool_calls]


def evaluate_tool_permission(tool_call: dict[str, Any]) -> ToolPermissionDecision:
    tool_name = str(tool_call.get("name") or "")
    tool_call_id = str(tool_call.get("id") or "")
    tool_args = tool_call.get("args") or {}

    for rule in PERMISSION_RULES:
        if _rule_applies(rule, tool_name) and rule["check"](tool_args):
            return _decision(
                tool_call_id,
                tool_name,
                rule["action"],
                rule["message"],
                rule["risk_summary"](tool_args),
            )

    return _decision(
        tool_call_id,
        tool_name,
        "allow",
        "No permission rule matched; allowed by default.",
        "",
    )


def approval_payload(decisions: list[ToolPermissionDecision]) -> dict[str, Any]:
    return {
        "type": "tool_approval",
        "message": "Approve these tool calls?",
        "tool_calls": [
            {
                "tool_call_id": decision["tool_call_id"],
                "tool_name": decision["tool_name"],
                "reason": decision["reason"],
                "risk_summary": decision["risk_summary"],
            }
            for decision in decisions
            if decision["action"] == "approval_required"
        ],
    }


def apply_approval_response(
    decisions: list[ToolPermissionDecision],
    approval_response: Any,
) -> list[ToolPermissionDecision]:
    approved = _approval_response_is_approved(approval_response)
    final_action: PermissionAction = "allow" if approved else "rejected"
    reason = "Approved by user." if approved else "Rejected by user."

    updated_decisions = []
    for decision in decisions:
        if decision["action"] != "approval_required":
            updated_decisions.append(decision)
            continue

        updated = dict(decision)
        updated["action"] = final_action
        updated["reason"] = reason
        updated_decisions.append(updated)

    return updated_decisions


def decision_by_tool_call_id(
    decisions: list[ToolPermissionDecision],
) -> dict[str, ToolPermissionDecision]:
    return {decision["tool_call_id"]: decision for decision in decisions}


def _approval_response_is_approved(approval_response: Any) -> bool:
    if isinstance(approval_response, dict):
        return bool(approval_response.get("approved"))
    return bool(approval_response)


def _decision(
    tool_call_id: str,
    tool_name: str,
    action: PermissionAction,
    reason: str,
    risk_summary: str,
) -> ToolPermissionDecision:
    return {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "action": action,
        "reason": reason,
        "risk_summary": risk_summary,
    }


def _rule_applies(rule: PermissionRule, tool_name: str) -> bool:
    return "*" in rule["tools"] or tool_name in rule["tools"]
