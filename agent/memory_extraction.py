import json
import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from agent.memory_namespaces import (
    MemoryCategory,
    memory_categories,
    namespace_for_category,
)
from agent.model import BASE_MODEL
from agent.state import AgentState, RuntimeContext
from snip_compact import serialize_message


logger = logging.getLogger("agent")

RECENT_USER_TURNS = 10
MAX_MESSAGE_CONTENT_CHARS = 4_000
MAX_EXTRACTION_CONTEXT_CHARS = 40_000
MAX_EXISTING_MEMORIES = 50
MEMORY_SOURCE = "extract_memory"

class ExtractedMemory(BaseModel):
    category: MemoryCategory = Field(
        description="Memory category determining where the item should be stored."
    )
    key: str = Field(
        description="Short snake_case key. Reuse an existing key when updating a memory."
    )
    summary: str = Field(
        description="A short directory-style description of this memory."
    )
    content: str = Field(
        description="Detailed durable memory content worth remembering for future work."
    )


class MemoryExtraction(BaseModel):
    memories: list[ExtractedMemory] = Field(
        default_factory=list,
        description="Memories to insert or update. Empty when nothing is worth saving.",
    )


def extract_long_term_memories(
    state: AgentState,
    runtime: Runtime[RuntimeContext],
) -> set[MemoryCategory]:
    if runtime.store is None:
        logger.info("Skipping memory extraction; runtime store is unavailable")
        return set()

    messages = recent_user_turn_window(state.get("messages") or [])
    if not messages:
        logger.info("Skipping memory extraction; no recent user messages")
        return set()

    try:
        existing_memories = load_existing_memories(runtime)
        extraction = run_memory_extraction(messages, existing_memories)
        stored_categories = store_extracted_memories(runtime, extraction)
        logger.info(
            "Long-term memory extraction touched categories: %s",
            sorted(stored_categories),
        )
        return stored_categories
    except Exception:
        logger.exception("Long-term memory extraction failed; continuing without blocking")
        return set()


def extract_project_memories(
    state: AgentState,
    runtime: Runtime[RuntimeContext],
) -> set[MemoryCategory]:
    return extract_long_term_memories(state, runtime)


def recent_user_turn_window(
    messages: list[BaseMessage],
    max_user_turns: int = RECENT_USER_TURNS,
) -> list[BaseMessage]:
    user_indexes = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    ]
    if not user_indexes:
        return []

    if len(user_indexes) <= max_user_turns:
        start_index = user_indexes[0]
    else:
        start_index = user_indexes[-max_user_turns]
    return messages[start_index:]


def run_memory_extraction(
    messages: list[BaseMessage],
    existing_memories: list[dict],
) -> MemoryExtraction:
    structured_model = BASE_MODEL.with_structured_output(MemoryExtraction)
    response = structured_model.invoke(
        [
            SystemMessage(content=MEMORY_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {
                        "existing_memories": existing_memories,
                        "recent_conversation": compact_messages_for_extraction(
                            messages
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            ),
        ]
    )
    if isinstance(response, MemoryExtraction):
        return response
    return MemoryExtraction.model_validate(response)


def store_extracted_memories(
    runtime: Runtime[RuntimeContext],
    extraction: MemoryExtraction,
) -> set[MemoryCategory]:
    if runtime.store is None:
        return set()

    stored_categories = set()
    for memory in extraction.memories:
        key = normalize_memory_key(memory.key)
        content = memory.content.strip()
        if not key or not content:
            continue

        namespace = namespace_for_category(memory.category, runtime.context)
        runtime.store.put(
            namespace,
            key,
            {
                "summary": memory.summary.strip() or content[:160],
                "detail": content,
                "fact": content,
                "source": MEMORY_SOURCE,
                "category": memory.category,
            },
        )
        stored_categories.add(memory.category)
    return stored_categories


def load_existing_memories(runtime: Runtime[RuntimeContext]) -> list[dict]:
    if runtime.store is None:
        return []

    memories = []
    for category in memory_categories():
        namespace = namespace_for_category(category, runtime.context)
        for item in runtime.store.search(namespace, limit=MAX_EXISTING_MEMORIES):
            memories.append(serialize_store_item(category, item))
    return memories


def serialize_store_item(category: MemoryCategory, item: Any) -> dict:
    return {
        "category": category,
        "namespace": getattr(item, "namespace", None),
        "key": getattr(item, "key", None),
        "value": getattr(item, "value", None),
    }


def compact_messages_for_extraction(messages: list[BaseMessage]) -> list[dict]:
    compacted_reversed = []
    total_chars = 0
    truncated = False
    for message in reversed(messages):
        data = serialize_message(message)
        content = str(data.get("content") or "")
        truncated_content = truncate_text(content, MAX_MESSAGE_CONTENT_CHARS)
        data["content"] = truncated_content
        data["content_length"] = len(content)

        data_chars = len(json.dumps(data, ensure_ascii=False, default=str))
        if total_chars + data_chars > MAX_EXTRACTION_CONTEXT_CHARS:
            truncated = True
            break

        compacted_reversed.append(data)
        total_chars += data_chars

    compacted = list(reversed(compacted_reversed))
    if truncated:
        compacted.insert(
            0,
            {
                "type": "system",
                "content": "[older memory extraction context truncated]",
            },
        )
    return compacted


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}...[truncated {omitted} chars]"


def normalize_memory_key(key: str) -> str:
    normalized = key.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized[:80]


MEMORY_EXTRACTION_SYSTEM_PROMPT = """You extract durable long-term memory.

Save only memories that will help future development work. Categorize each
memory carefully:
- user_preferences: stable user preferences, such as language, style, workflow,
  explanation depth, testing preferences, or abstraction preferences.
- user_feedback: feedback correcting agent behavior or preferences learned from
  user reactions.
- project_facts: durable facts about the project, architecture, tools, or current
  implementation.
- project_decisions: confirmed decisions and the chosen direction for this
  project.
- project_references: useful paths, docs, APIs, links, or reference material.
- project_open_threads: unresolved follow-ups or active threads that should be
  remembered for future work.

Do not save:
- Casual conversation.
- One-off errors or transient command output.
- Ordinary completed steps with no durable value.
- Secrets, credentials, API keys, tokens, private keys, or sensitive personal data.

Use existing_memories to avoid duplicates. If a new memory updates an existing
memory, reuse the existing category and key. If nothing is worth remembering,
return an empty memories list. Keys must be concise snake_case.

For each memory:
- summary is a short catalog description, usually one sentence.
- content is the concrete detail to load when that catalog item is relevant.
"""
