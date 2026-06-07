import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from agent.memory_extraction import normalize_memory_key
from agent.memory_namespaces import MemoryCategory, namespace_for_category
from agent.model import BASE_MODEL
from agent.state import RuntimeContext


logger = logging.getLogger("agent")

MAX_CONSOLIDATION_MEMORIES = 80
MEMORY_SOURCE = "memory_consolidation"


class ConsolidatedMemory(BaseModel):
    key: str = Field(description="Existing or new snake_case key to upsert.")
    summary: str = Field(description="Short catalog description.")
    detail: str = Field(description="Consolidated detailed memory content.")


class MemoryDeletion(BaseModel):
    key: str = Field(description="Existing key to delete.")
    reason: str = Field(description="Why this memory should be deleted.")


class MemoryConsolidation(BaseModel):
    upserts: list[ConsolidatedMemory] = Field(
        default_factory=list,
        description="Memories to create or overwrite after consolidation.",
    )
    deletes: list[MemoryDeletion] = Field(
        default_factory=list,
        description="Existing memories to delete because they are stale, duplicated, or low-value.",
    )


def consolidate_long_term_memories(
    runtime: Runtime[RuntimeContext],
    categories: set[MemoryCategory],
) -> None:
    if runtime.store is None:
        logger.info("Skipping memory consolidation; runtime store is unavailable")
        return

    for category in sorted(categories):
        try:
            consolidate_memory_category(runtime, category)
        except Exception:
            logger.exception(
                "Memory consolidation failed for category %s; continuing",
                category,
            )


def consolidate_memory_category(
    runtime: Runtime[RuntimeContext],
    category: MemoryCategory,
) -> None:
    if runtime.store is None:
        return

    namespace = namespace_for_category(category, runtime.context)
    items = list(runtime.store.search(namespace, limit=MAX_CONSOLIDATION_MEMORIES))
    if len(items) < 2:
        logger.info(
            "Skipping memory consolidation for %s; only %s item(s)",
            category,
            len(items),
        )
        return

    consolidation = run_memory_consolidation(category, serialize_items(items))
    apply_memory_consolidation(runtime, category, consolidation)
    logger.info(
        "Memory consolidation for %s applied %s upserts and %s deletes",
        category,
        len(consolidation.upserts),
        len(consolidation.deletes),
    )


def run_memory_consolidation(
    category: MemoryCategory,
    memories: list[dict],
) -> MemoryConsolidation:
    structured_model = BASE_MODEL.with_structured_output(MemoryConsolidation)
    response = structured_model.invoke(
        [
            SystemMessage(content=MEMORY_CONSOLIDATION_SYSTEM_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {
                        "category": category,
                        "memories": memories,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            ),
        ]
    )
    if isinstance(response, MemoryConsolidation):
        return response
    return MemoryConsolidation.model_validate(response)


def apply_memory_consolidation(
    runtime: Runtime[RuntimeContext],
    category: MemoryCategory,
    consolidation: MemoryConsolidation,
) -> None:
    if runtime.store is None:
        return

    namespace = namespace_for_category(category, runtime.context)
    for deletion in consolidation.deletes:
        key = normalize_memory_key(deletion.key)
        if key:
            runtime.store.delete(namespace, key)

    for memory in consolidation.upserts:
        key = normalize_memory_key(memory.key)
        detail = memory.detail.strip()
        if not key or not detail:
            continue

        runtime.store.put(
            namespace,
            key,
            {
                "summary": memory.summary.strip() or detail[:160],
                "detail": detail,
                "fact": detail,
                "source": MEMORY_SOURCE,
                "category": category,
            },
        )


def serialize_items(items: list[Any]) -> list[dict]:
    serialized = []
    for item in items:
        serialized.append(
            {
                "key": getattr(item, "key", None),
                "value": getattr(item, "value", None),
            }
        )
    return serialized


MEMORY_CONSOLIDATION_SYSTEM_PROMPT = """You maintain a long-term memory store.

You receive all memories from one category. Clean them up so the memory store
stays fresh, accurate, and useful.

Actions:
- Merge duplicates into one concise memory.
- Update stale or conflicting memories to the newest confirmed version.
- Delete low-value, completed, obsolete, or duplicate memories.
- Keep useful stable preferences, durable feedback, active decisions, references,
  facts, and unresolved open threads.

Category-specific guidance:
- project_open_threads: remove threads that are already completed or converted
  into durable facts/decisions.
- project_decisions: prefer the latest confirmed decision when conflicts exist.
- user_feedback: keep behavioral corrections that should affect future agent work.
- user_preferences: keep stable preferences only.

Return only the upserts and deletes needed. If the category is already clean,
return empty lists. Do not invent new memories unrelated to the provided items.
"""
