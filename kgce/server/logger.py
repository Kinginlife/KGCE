 
import logging

uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.INFO)

kgce_logger = logging.getLogger("kgce-server")
kgce_logger.setLevel(logging.INFO)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "format": "[%(asctime)s %(process)d:%(threadName)s] %(name)s - "
            "%(levelname)s - %(message)s | %(filename)s:%(lineno)d",
        },
        "logformat": {
            "format": "[%(asctime)s %(process)d:%(threadName)s] %(name)s - "
            "%(levelname)s - %(message)s | %(filename)s:%(lineno)d"
        },
    },
    "handlers": {
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "logformat",
            "filename": "info.log",
            "encoding": "utf8",
            "mode": "a",
        },
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["default", "file_handler"],
            "propagate": False,
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["default", "file_handler"],
        "propagate": False,
    },
}
