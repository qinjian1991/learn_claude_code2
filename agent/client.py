from collections.abc import Iterator
import logging

from core.config import settings
from core.constants import WORKDIR
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Command

from agent.graph import GRAPH
from agent.memory_namespaces import memory_categories, namespace_for_category
from agent.persistence import CHECKPOINTER, STORE
from agent.state import RuntimeContext, state_snapshot
from thread_memory import get_or_create_default_thread_id, rotate_default_thread_id


logger = logging.getLogger("agent")


class Agent:
    def __init__(self, thread_id: str | None = None) -> None:
        self.thread_id = thread_id or get_or_create_default_thread_id()
        logger.info("Agent initialized; thread_id: %s", self.thread_id)

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    @property
    def runtime_context(self) -> RuntimeContext:
        return {
            "workspace": str(WORKDIR),
            "user_id": settings.user_id,
            "project_id": settings.project_id,
        }

    def get_messages(self) -> list[BaseMessage]:
        return list(self.get_state_values().get("messages") or [])

    def get_state_values(self) -> dict:
        state = GRAPH.get_state(self.config)
        return dict(state.values if state else {})

    def stream(self, message: str) -> Iterator[str]:
        state_values = self.get_state_values()
        state_meta = state_snapshot(state_values)
        messages = list(state_values.get("messages") or [])
        messages.append(HumanMessage(content=message))
        logger.info("User message received; message length: %s", len(message))
        yield from self._stream_graph_input(
            {
                **state_meta,
                "messages": messages,
                "conversation_turn": state_meta["conversation_turn"] + 1,
                "last_context_action": "user_message",
            }
        )

    def resume_tool_approval(self, approved: bool) -> Iterator[str]:
        logger.info("Resuming tool approval; approved=%s", approved)
        yield from self._stream_graph_input(
            Command(resume={"approved": approved})
        )

    def get_pending_interrupt(self) -> dict | None:
        state = GRAPH.get_state(self.config)
        interrupts = getattr(state, "interrupts", None) or ()
        if not interrupts:
            return None

        value = interrupts[0].value
        if isinstance(value, dict):
            return value

        return {"type": "unknown", "message": str(value), "tool_calls": []}

    def _stream_graph_input(self, graph_input) -> Iterator[str]:
        with GRAPH.stream_events(
            graph_input,
            self.config,
            context=self.runtime_context,
            version="v3",
        ) as run:
            for message_stream in run.messages:
                if message_stream.node != "invoke_model":
                    continue
                yield from message_stream.text

            output = run.output
            if output:
                if "__interrupt__" in output:
                    logger.info("Agent stream interrupted for approval.")
                elif "messages" in output:
                    logger.info(
                        "Agent stream completed; stored messages: %s",
                        len(output["messages"]),
                    )

    def reset(self) -> None:
        CHECKPOINTER.delete_thread(self.thread_id)
        logger.info("Agent reset; thread_id: %s", self.thread_id)

    def new_thread(self) -> str:
        self.thread_id = rotate_default_thread_id()
        logger.info("Agent switched to new thread; thread_id: %s", self.thread_id)
        return self.thread_id

    def remember_user_preference(self, key: str, value: dict) -> None:
        STORE.put(("users", settings.user_id, "preferences"), key, value)
        logger.info("Stored user preference memory: %s", key)

    def remember_project_fact(self, key: str, value: dict) -> None:
        STORE.put(("projects", settings.project_id, "facts"), key, value)
        logger.info("Stored project fact memory: %s", key)

    def get_long_term_memories(self) -> list[dict]:
        items = []
        for category in memory_categories():
            namespace = namespace_for_category(category, self.runtime_context)
            items.extend(STORE.search(namespace, limit=50))

        return [
            {
                "namespace": item.namespace,
                "key": item.key,
                "value": item.value,
            }
            for item in items
        ]
