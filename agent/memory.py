import json
import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from agent.memory_namespaces import memory_categories, namespace_for_category
from agent.model import BASE_MODEL
from agent.state import RuntimeContext


logger = logging.getLogger("agent")
MEMORY_CANDIDATE_LIMIT_PER_CATEGORY = 20
SELECTED_MEMORY_LIMIT = 12


class SelectedMemoryReference(BaseModel):
    category: str = Field(description="Memory category from the catalog.")
    key: str = Field(description="Memory key from the catalog.")


class MemorySelection(BaseModel):
    selected: list[SelectedMemoryReference] = Field(
        default_factory=list,
        description="Catalog entries whose details should be loaded.",
    )


def load_long_term_memories(
    runtime: Runtime[RuntimeContext],
    messages: list[BaseMessage] | None = None,
) -> list[str]:
    if runtime.store is None:
        return []

    memory_items = load_memory_items(runtime)
    if not memory_items:
        return []

    selected_refs = select_memory_details(memory_items, messages or [])
    memories = format_selected_memories(memory_items, selected_refs)
    if memories:
        logger.info(
            "Loaded long-term memories: catalog=%s selected_details=%s",
            len(memory_items),
            len(selected_refs),
        )
    return memories


def load_memory_items(runtime: Runtime[RuntimeContext]) -> list[dict]:
    memory_items = []
    for category in memory_categories():
        namespace = namespace_for_category(category, runtime.context)
        for item in runtime.store.search(
            namespace,
            limit=MEMORY_CANDIDATE_LIMIT_PER_CATEGORY,
        ):
            memory_items.append(serialize_memory_item(category, item))
    return memory_items


def select_memory_details(
    memory_items: list[dict],
    messages: list[BaseMessage],
) -> set[tuple[str, str]]:
    try:
        structured_model = BASE_MODEL.with_structured_output(MemorySelection)
        response = structured_model.invoke(
            [
                SystemMessage(content=MEMORY_SELECTION_SYSTEM_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        {
                            "catalog": [
                                {
                                    "category": item["category"],
                                    "key": item["key"],
                                    "summary": item["summary"],
                                }
                                for item in memory_items
                            ],
                            "recent_conversation": serialize_recent_messages(messages),
                            "limit": SELECTED_MEMORY_LIMIT,
                        },
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
                ),
            ]
        )
        if not isinstance(response, MemorySelection):
            response = MemorySelection.model_validate(response)
        return {
            (item.category, item.key)
            for item in response.selected[:SELECTED_MEMORY_LIMIT]
        }
    except Exception:
        logger.exception("Long-term memory selection failed; loading catalog only")
        return set()


def serialize_recent_messages(messages: list[BaseMessage], limit: int = 8) -> list[dict]:
    recent = messages[-limit:]
    return [
        {
            "type": message.type,
            "content": str(message.content)[:2_000],
        }
        for message in recent
    ]


def format_selected_memories(
    memory_items: list[dict],
    selected_refs: set[tuple[str, str]],
) -> list[str]:
    memories = ["Catalog:"]
    for item in memory_items:
        memories.append(format_memory_catalog_item(item))

    selected_items = [
        item
        for item in memory_items
        if (item["category"], item["key"]) in selected_refs
    ]
    if selected_items:
        memories.append("")
        memories.append("Selected details:")
        memories.extend(format_memory_detail_item(item) for item in selected_items)

    return memories


def serialize_memory_item(category: str, item: Any) -> dict:
    value = getattr(item, "value", None) or {}
    if not isinstance(value, dict):
        value = {"detail": value}

    detail = str(value.get("detail") or value.get("fact") or "")
    summary = str(value.get("summary") or detail[:160])
    namespace = getattr(item, "namespace", ())
    key = str(getattr(item, "key", ""))
    return {
        "category": category,
        "namespace": namespace,
        "key": key,
        "summary": summary,
        "detail": detail,
        "value": value,
    }


def format_memory_catalog_item(item: dict) -> str:
    namespace = "/".join(item["namespace"])
    return f"- [{item['category']}:{namespace}/{item['key']}] {item['summary']}"


def format_memory_detail_item(item: dict) -> str:
    namespace = "/".join(item["namespace"])
    payload = {
        "summary": item["summary"],
        "detail": item["detail"],
        "metadata": {
            key: value
            for key, value in item["value"].items()
            if key not in {"summary", "detail", "fact"}
        },
    }
    return f"- [{item['category']}:{namespace}/{item['key']}] {json.dumps(payload, ensure_ascii=False)}"


MEMORY_SELECTION_SYSTEM_PROMPT = """You select long-term memory details to load.

You will receive a catalog of memories with category, key, and summary. Select
only the memories whose full details are likely useful for the next model call.

Prefer stable user preferences and feedback, active project decisions, relevant
project facts, useful references, and open threads. Return at most the requested
limit. If no details are needed, return an empty list.
"""
