from pathlib import Path
from uuid import uuid4

from core.constants import WORKDIR
from core.config import settings


DEFAULT_THREAD_ID_PATH = "logs/current_thread_id.txt"


def get_thread_id_path() -> Path:
    return WORKDIR / getattr(settings, "thread_id_path", DEFAULT_THREAD_ID_PATH)


def get_or_create_default_thread_id() -> str:
    thread_id = read_default_thread_id()
    if thread_id:
        return thread_id

    return create_default_thread_id()


def create_default_thread_id() -> str:
    thread_id = f"local-{uuid4()}"
    write_default_thread_id(thread_id)
    return thread_id


def read_default_thread_id() -> str | None:
    thread_id_path = get_thread_id_path()
    if not thread_id_path.exists():
        return None

    thread_id = thread_id_path.read_text(encoding="utf-8").strip()
    return thread_id or None


def write_default_thread_id(thread_id: str) -> None:
    thread_id_path = get_thread_id_path()
    thread_id_path.parent.mkdir(parents=True, exist_ok=True)
    thread_id_path.write_text(thread_id, encoding="utf-8")


def rotate_default_thread_id() -> str:
    return create_default_thread_id()
