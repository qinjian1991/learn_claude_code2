import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from agent.prompt import build_system_prompt
from agent.task_store import (
    claim_task,
    complete_task,
    create_task,
    load_task_graph,
    render_task_graph_summary,
    save_task_graph,
    task_graph_json,
)
from agent.tools.task_tools import task_claim, task_complete, task_create, task_list
from core.constants import WORKDIR


class TaskStoreTest(unittest.TestCase):
    def test_missing_graph_initializes_empty_file(self):
        with tempfile.TemporaryDirectory(dir=WORKDIR) as temp_dir:
            path = Path(temp_dir) / "tasks.json"

            graph = load_task_graph(path)

            self.assertEqual(graph, {"version": 1, "tasks": []})
            self.assertTrue(path.exists())

    def test_create_task_writes_json(self):
        with tempfile.TemporaryDirectory(dir=WORKDIR) as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            task = create_task(
                "Plan",
                "Plan the work",
                path=path,
                now=datetime(2026, 6, 18, 9, 30, 0),
            )

            graph = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(task["id"], "task_20260618_001")
            self.assertEqual(graph["tasks"][0]["title"], "Plan")
            self.assertFalse(path.with_name("tasks.json.tmp").exists())

    def test_create_task_rejects_missing_dependency(self):
        with tempfile.TemporaryDirectory(dir=WORKDIR) as temp_dir:
            path = Path(temp_dir) / "tasks.json"

            with self.assertRaisesRegex(ValueError, "Unknown dependency"):
                create_task("Build", "Build it", ["missing"], path=path)

    def test_save_task_graph_rejects_cycle(self):
        with tempfile.TemporaryDirectory(dir=WORKDIR) as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            graph = {
                "version": 1,
                "tasks": [
                    {
                        "id": "a",
                        "title": "A",
                        "description": "",
                        "depends_on": ["b"],
                        "status": "pending",
                    },
                    {
                        "id": "b",
                        "title": "B",
                        "description": "",
                        "depends_on": ["a"],
                        "status": "pending",
                    },
                ],
            }

            with self.assertRaisesRegex(ValueError, "cycle"):
                save_task_graph(graph, path)


class TaskTransitionTest(unittest.TestCase):
    def test_pending_claimed_completed_lifecycle(self):
        with tempfile.TemporaryDirectory(dir=WORKDIR) as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            root = create_task("Root", "Root task", path=path)

            claimed = claim_task(root["id"], "agent:test", path=path)
            completed = complete_task(root["id"], "done", path=path)

            self.assertEqual(claimed["status"], "claimed")
            self.assertEqual(claimed["owner"], "agent:test")
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["result"], "done")

    def test_dependency_must_be_completed_before_claim(self):
        with tempfile.TemporaryDirectory(dir=WORKDIR) as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            root = create_task("Root", "Root task", path=path)
            child = create_task("Child", "Child task", [root["id"]], path=path)

            with self.assertRaisesRegex(ValueError, "dependencies complete"):
                claim_task(child["id"], "agent:test", path=path)

            claim_task(root["id"], "agent:test", path=path)
            complete_task(root["id"], "done", path=path)
            claimed_child = claim_task(child["id"], "agent:test", path=path)

            self.assertEqual(claimed_child["status"], "claimed")

    def test_completed_task_cannot_be_claimed_again(self):
        with tempfile.TemporaryDirectory(dir=WORKDIR) as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            task = create_task("Task", "Task", path=path)
            claim_task(task["id"], "agent:test", path=path)
            complete_task(task["id"], "done", path=path)

            with self.assertRaisesRegex(ValueError, "cannot be claimed"):
                claim_task(task["id"], "agent:test", path=path)

    def test_pending_task_cannot_be_completed(self):
        with tempfile.TemporaryDirectory(dir=WORKDIR) as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            task = create_task("Task", "Task", path=path)

            with self.assertRaisesRegex(ValueError, "cannot be completed"):
                complete_task(task["id"], "done", path=path)


class TaskToolTest(unittest.TestCase):
    def test_task_tools_create_claim_complete_and_list(self):
        with tempfile.TemporaryDirectory(dir=WORKDIR) as temp_dir:
            path = str(Path(temp_dir) / "tasks.json")
            with patch("agent.task_store.settings.task_graph_path", path):
                create_result = task_create.invoke(
                    {
                        "title": "Plan",
                        "description": "Plan the work",
                        "depends_on": [],
                    }
                )
                task_id = create_result.split()[2].rstrip(":")
                claim_result = task_claim.invoke({"task_id": task_id})
                complete_result = task_complete.invoke(
                    {"task_id": task_id, "result": "finished"}
                )
                listed = task_list.invoke({"compact": False})

            self.assertIn("Created task", create_result)
            self.assertIn("Claimed task", claim_result)
            self.assertIn("Completed task", complete_result)
            self.assertIn("finished", listed)


class TaskPromptTest(unittest.TestCase):
    def test_empty_task_graph_summary(self):
        with tempfile.TemporaryDirectory(dir=WORKDIR) as temp_dir:
            path = Path(temp_dir) / "tasks.json"

            summary = render_task_graph_summary(path)

            self.assertIn("TASK_GRAPH:", summary)
            self.assertIn("Total tasks: 0", summary)
            self.assertIn("Ready to claim:", summary)

    def test_summary_shows_ready_and_claimed_tasks(self):
        with tempfile.TemporaryDirectory(dir=WORKDIR) as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            ready = create_task("Ready", "Ready", path=path)
            claimed = create_task("Claimed", "Claimed", path=path)
            claim_task(claimed["id"], "agent:test", path=path)

            summary = render_task_graph_summary(path)

            self.assertIn(ready["id"], summary)
            self.assertIn(claimed["id"], summary)
            self.assertIn("claimed=1", summary)

    def test_summary_truncates_and_mentions_task_list(self):
        with tempfile.TemporaryDirectory(dir=WORKDIR) as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            for index in range(10):
                create_task(f"Task {index}", "Task", path=path)

            summary = render_task_graph_summary(path, max_items=3)

            self.assertIn("call task_list", summary)

    @patch("agent.prompt.build_skill_directory_section", return_value="Skills: none")
    def test_system_prompt_includes_task_graph(self, _skills):
        with patch(
            "agent.prompt.render_task_graph_summary",
            return_value="TASK_GRAPH:\nTotal tasks: 0",
        ):
            prompt = build_system_prompt(
                {"workspace": "E:/work", "user_id": "u", "project_id": "p"}
            )

        self.assertIn("TASK_GRAPH:", prompt)


if __name__ == "__main__":
    unittest.main()
