import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from agent.nodes import call_tools, check_tool_permissions
from agent.tools.permissions import PERMISSION_RULES, evaluate_tool_permission


def tool_call(name, args=None, call_id="call_test"):
    return {
        "name": name,
        "args": args or {},
        "id": call_id,
        "type": "tool_call",
    }


class FakeTool:
    def __init__(self, result):
        self.result = result

    def invoke(self, args):
        return self.result


class ToolPermissionPolicyTest(unittest.TestCase):
    def test_permission_rules_are_centrally_registered(self):
        registered_tools = {
            tool
            for rule in PERMISSION_RULES
            for tool in rule["tools"]
        }

        self.assertIn("write_file", registered_tools)
        self.assertIn("edit_file", registered_tools)
        self.assertIn("powershell", registered_tools)
        self.assertNotIn("read_file", registered_tools)
        self.assertNotIn("glob", registered_tools)
        self.assertNotIn("load_skill", registered_tools)
        self.assertNotIn("run_todo_write", registered_tools)
        self.assertNotIn("task_list", registered_tools)
        self.assertNotIn("task_create", registered_tools)
        self.assertNotIn("task_claim", registered_tools)
        self.assertNotIn("task_complete", registered_tools)

    def test_read_only_tools_are_allowed(self):
        for name in (
            "read_file",
            "glob",
            "load_skill",
            "run_todo_write",
            "task_list",
            "task_create",
            "task_claim",
            "task_complete",
        ):
            decision = evaluate_tool_permission(tool_call(name))
            self.assertEqual(decision["action"], "allow")

    def test_write_file_and_edit_file_require_approval(self):
        for name in ("write_file", "edit_file"):
            decision = evaluate_tool_permission(
                tool_call(name, {"path": "sample.txt"})
            )
            self.assertEqual(decision["action"], "approval_required")

    def test_destructive_git_commands_are_denied(self):
        for command in ("git reset --hard", "git clean -fdx"):
            decision = evaluate_tool_permission(
                tool_call("powershell", {"command": command})
            )
            self.assertEqual(decision["action"], "deny")

    def test_read_only_shell_commands_are_allowed(self):
        for command in ("git status", "git diff", "git log", "git show HEAD", "rg test"):
            decision = evaluate_tool_permission(
                tool_call("powershell", {"command": command})
            )
            self.assertEqual(decision["action"], "allow")


class ToolPermissionNodeTest(unittest.TestCase):
    def test_denied_tool_call_does_not_interrupt(self):
        message = AIMessage(
            content="",
            tool_calls=[
                tool_call("powershell", {"command": "git reset --hard"}, "call_deny")
            ],
        )
        result = check_tool_permissions({"messages": [message]})

        self.assertEqual(result["tool_permission_decisions"][0]["action"], "deny")

    def test_approval_resume_approve_allows_tool(self):
        message = AIMessage(
            content="",
            tool_calls=[
                tool_call("write_file", {"path": "sample.txt", "content": "x"}, "call_write")
            ],
        )

        with patch("agent.nodes.interrupt", return_value={"approved": True}):
            result = check_tool_permissions({"messages": [message]})

        self.assertEqual(result["tool_permission_decisions"][0]["action"], "allow")

    def test_approval_resume_reject_blocks_tool(self):
        message = AIMessage(
            content="",
            tool_calls=[
                tool_call("write_file", {"path": "sample.txt", "content": "x"}, "call_write")
            ],
        )

        with patch("agent.nodes.interrupt", return_value={"approved": False}):
            result = check_tool_permissions({"messages": [message]})

        self.assertEqual(result["tool_permission_decisions"][0]["action"], "rejected")

    def test_mixed_tool_calls_execute_allowed_and_report_denied(self):
        message = AIMessage(
            content="",
            tool_calls=[
                tool_call("read_file", {"path": "sample.txt"}, "call_read"),
                tool_call("powershell", {"command": "git reset --hard"}, "call_deny"),
            ],
        )
        state = {
            "messages": [message],
            "tool_permission_decisions": [
                {
                    "tool_call_id": "call_read",
                    "tool_name": "read_file",
                    "action": "allow",
                    "reason": "Read-only workspace tool.",
                    "risk_summary": "",
                },
                {
                    "tool_call_id": "call_deny",
                    "tool_name": "powershell",
                    "action": "deny",
                    "reason": "Command is in the deny list.",
                    "risk_summary": "git reset --hard",
                },
            ],
        }

        with patch.dict("agent.nodes.TOOLS_BY_NAME", {"read_file": FakeTool("READ")}):
            result = call_tools(state)

        tool_messages = result["messages"][-2:]
        self.assertEqual(tool_messages[0].content, "READ")
        self.assertIn("deny", tool_messages[1].content)


if __name__ == "__main__":
    unittest.main()
