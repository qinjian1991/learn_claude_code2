from agent.state import RuntimeContext
from agent.skills import build_skill_directory_section
from agent.todos import render_current_todos


PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "context_compaction": (
        "If the conversation context becomes noisy, repetitive, or hard to reason about, "
        "call request_panorama_summary to compact the full conversation before continuing."
    ),
}


def build_system_prompt(
    runtime_context: RuntimeContext,
    long_term_memories: list[str] | None = None,
    current_todos: list[dict] | None = None,
) -> str:
    sections = []

    sections.append(PROMPT_SECTIONS["identity"])
    sections.append(f"Working directory: {runtime_context['workspace']}")
    sections.append(render_current_todos(current_todos))
    sections.append(build_skill_directory_section())
    if long_term_memories:
        sections.append("Long-term memory:\n" + "\n".join(long_term_memories))
    sections.append(PROMPT_SECTIONS["context_compaction"])

    return "\n\n".join(sections)
