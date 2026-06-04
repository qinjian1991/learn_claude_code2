from logging.config import dictConfig

from core.constants import WORKDIR


LOG_DIR = WORKDIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "context_debug": {
            "format": "%(asctime)s | %(levelname)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        },
        "agent_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "default",
            "filename": str(LOG_DIR / "agent.log"),
            "maxBytes": 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf-8",
        },
        "agent_context_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "context_debug",
            "filename": str(LOG_DIR / "agent_context_debug.log"),
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 10,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "agent": {
            "level": "INFO",
            "handlers": ["console", "agent_file"],
            "propagate": False,
        },
        "agent.context": {
            "level": "DEBUG",
            "handlers": ["agent_context_file"],
            "propagate": False,
        },
    },
}


def configure_logging() -> None:
    dictConfig(LOGGING_CONFIG)
