import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from agent.nodes import call_tools
from agent.prompt import build_system_prompt
from agent.state import state_snapshot
from agent.todos import normalize_todos, render_current_todos, todo_update_summary


def tool_call(name, args=None, call_id="call_test"):
    return {
        "name": name,
        "args": args or {},
        "id": call_id,
        "type": "tool_call",
    }


class TodoValidationTest(unittest.TestCase):
    def test_accepts_valid_full_replacement(self):
        todos = normalize_todos(
            [
                {"content": "Plan work", "status": "pending"},
                {"content": "Implement", "status": "in_progress"},
            ]
        )

        self.assertEqual(
            todos,
            [
                {"content": "Plan work", "status": "pending"},
                {"content": "Implement", "status": "in_progress"},
            ],
        )

    def test_rejects_missing_content(self):
        with self.assertRaisesRegex(ValueError, "missing content"):
            normalize_todos([{"content": "  ", "status": "pending"}])

    def test_rejects_unknown_status(self):
        with self.assertRaisesRegex(ValueError, "invalid status"):
            normalize_todos([{"content": "Plan", "status": "started"}])

    def test_normalizes_whitespace(self):
        todos = normalize_todos(
            [{"content": "  Plan   the   work\nnow ", "status": "PENDING"}]
        )

        self.assertEqual(todos[0]["content"], "Plan the work now")
        self.assertEqual(todos[0]["status"], "pending")

    def test_update_summary_counts_statuses(self):
        summary = todo_update_summary(
            [
                {"content": "A", "status": "pending"},
                {"content": "B", "status": "completed"},
                {"content": "C", "status": "completed"},
            ]
        )

        self.assertIn("3 items", summary)
        self.assertIn("pending=1", summary)
        self.assertIn("completed=2", summary)


class TodoPromptTest(unittest.TestCase):
    def test_empty_todos_render_planning_reminder(self):
        section = render_current_todos([])

        self.assertIn("CURRENT_TODOS:", section)
        self.assertIn("(none)", section)
        self.assertIn("call run_todo_write", section)

    def test_non_empty_todos_render_statuses(self):
        section = render_current_todos(
            [{"content": "Implement planning", "status": "in_progress"}]
        )

        self.assertIn("1. [in_progress] Implement planning", section)

    @patch("agent.prompt.build_skill_directory_section", return_value="Skills: none")
    def test_system_prompt_includes_current_todos(self, _skills):
        prompt = build_system_prompt(
            {"workspace": "E:/work", "user_id": "u", "project_id": "p"},
            current_todos=[
                {"content": "Write tests", "status": "pending"},
            ],
        )

        self.assertIn("CURRENT_TODOS:", prompt)
        self.assertIn("[pending] Write tests", prompt)


class TodoToolNodeTest(unittest.TestCase):
    def test_run_todo_write_updates_agent_state_without_approval(self):
        message = AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    "run_todo_write",
                    {
                        "todos": [
                            {"content": "Plan", "status": "completed"},
                            {"content": "Build", "status": "in_progress"},
                        ]
                    },
                    "call_todo",
                )
            ],
        )
        state = {
            "messages": [message],
            "conversation_turn": 3,
            "todo_update_count": 1,
            "tool_permission_decisions": [
                {
                    "tool_call_id": "call_todo",
                    "tool_name": "run_todo_write",
                    "action": "allow",
                    "reason": "No permission rule matched; allowed by default.",
                    "risk_summary": "",
                }
            ],
        }

        result = call_tools(state)

        self.assertEqual(
            result["current_todos"],
            [
                {"content": "Plan", "status": "completed"},
                {"content": "Build", "status": "in_progress"},
            ],
        )
        self.assertEqual(result["todo_update_count"], 2)
        self.assertEqual(result["last_todo_update_turn"], 3)
        self.assertIn("completed=1", result["messages"][-1].content)
        self.assertIn("in_progress=1", result["messages"][-1].content)

    def test_invalid_todo_write_returns_tool_error_without_state_update(self):
        message = AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    "run_todo_write",
                    {"todos": [{"content": "", "status": "pending"}]},
                    "call_todo",
                )
            ],
        )
        state = {
            "messages": [message],
            "tool_permission_decisions": [
                {
                    "tool_call_id": "call_todo",
                    "tool_name": "run_todo_write",
                    "action": "allow",
                    "reason": "No permission rule matched; allowed by default.",
                    "risk_summary": "",
                }
            ],
        }

        result = call_tools(state)

        self.assertNotIn("current_todos", result)
        self.assertIn("Tool error:", result["messages"][-1].content)


class TodoStateTest(unittest.TestCase):
    def test_state_snapshot_preserves_todo_state(self):
        snapshot = state_snapshot(
            {
                "current_todos": [
                    {"content": "Keep planning state", "status": "pending"}
                ],
                "todo_update_count": 4,
                "last_todo_update_turn": 7,
            }
        )

        self.assertEqual(
            snapshot["current_todos"],
            [{"content": "Keep planning state", "status": "pending"}],
        )
        self.assertEqual(snapshot["todo_update_count"], 4)
        self.assertEqual(snapshot["last_todo_update_turn"], 7)


if __name__ == "__main__":
    unittest.main()
