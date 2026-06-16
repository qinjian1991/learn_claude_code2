import logging

from langchain_core.tools import tool

from agent.skills import load_skill_content


logger = logging.getLogger("agent")


@tool
def load_skill(skill_name: str) -> str:
    """Load the full SKILL.md instructions for a named local skill."""
    logger.info("Loading skill: %s", skill_name)
    return load_skill_content(skill_name)
