import unittest

from agent.prompt import build_system_prompt
from agent.skills import build_skill_directory_section, discover_skills, load_skill_content
from agent.tools.load_skill import load_skill


class SkillDiscoveryTest(unittest.TestCase):
    def test_discovers_installed_skills(self):
        names = {skill.name for skill in discover_skills()}

        self.assertIn("langchain-middleware", names)
        self.assertIn("langgraph-cli", names)

    def test_system_prompt_includes_skill_directory_and_instruction(self):
        prompt = build_system_prompt(
            {
                "workspace": "E:/work/learn_claude_code2",
                "user_id": "test-user",
                "project_id": "test-project",
            }
        )

        self.assertIn("Available skills:", prompt)
        self.assertIn("- langchain-middleware:", prompt)
        self.assertIn("load_skill", prompt)
        self.assertIn("full SKILL.md instructions", prompt)

    def test_load_skill_returns_full_skill_file(self):
        content = load_skill_content("langchain-middleware")

        self.assertIn("name: langchain-middleware", content)
        self.assertIn("<overview>", content)

    def test_load_skill_tool_returns_full_skill_file(self):
        content = load_skill.invoke({"skill_name": "langchain-middleware"})

        self.assertIn("name: langchain-middleware", content)
        self.assertIn("Human-in-the-Loop", content)

    def test_unknown_skill_reports_available_skills(self):
        content = load_skill_content("missing-skill")

        self.assertIn("Skill not found: missing-skill", content)
        self.assertIn("Available skills:", content)
        self.assertIn("langchain-middleware", content)

    def test_folded_multiline_description_is_included(self):
        section = build_skill_directory_section()

        self.assertIn(
            "- swarm: Dispatches many independent items in parallel: create a table, "
            "fan out to subagents, aggregate results. One row = one unit of work.",
            section,
        )


if __name__ == "__main__":
    unittest.main()
