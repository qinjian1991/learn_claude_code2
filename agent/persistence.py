import sqlite3

from core.config import settings
from core.constants import WORKDIR
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore


CHECKPOINT_DB_PATH = WORKDIR / settings.checkpoint_db_path
CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
CHECKPOINT_CONNECTION = sqlite3.connect(
    CHECKPOINT_DB_PATH,
    check_same_thread=False,
)
CHECKPOINTER = SqliteSaver(CHECKPOINT_CONNECTION)
CHECKPOINTER.setup()

STORE_DB_PATH = WORKDIR / settings.store_db_path
STORE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
STORE_CONNECTION = sqlite3.connect(
    STORE_DB_PATH,
    check_same_thread=False,
    isolation_level=None,
)
STORE = SqliteStore(STORE_CONNECTION)
STORE.setup()
