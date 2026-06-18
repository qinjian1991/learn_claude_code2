from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class TodoItem(BaseModel):
    content: str = Field(description="The TODO item text.")
    status: Literal["pending", "in_progress", "completed", "cancelled"] = Field(
        description="The current status of this TODO item."
    )


@tool
def run_todo_write(todos: list[TodoItem]) -> str:
    """Replace CURRENT_TODOS with the complete updated TODO list."""
    return "CURRENT_TODOS update accepted."
