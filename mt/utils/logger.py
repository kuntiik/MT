# import logging
from loguru import logger
import sys
from typing import Optional, Any

logger.level("AUTOFIX", 25)
logger.level("AUTOFIX-START", 25, color="<fg #1f6f8b><bold>")
logger.level("AUTOFIX-SUCCESS", 25, color="<green>")
logger.level("AUTOFIX-FAIL", 25, color="<fg #ea2c62>")
logger.level("AUTOFIX-REPORT", 25, color="<fg #9a7d0a>")


def autofix_log(
        level: str, message: str, *args, record_id: Optional[Any] = None
) -> None:
    if record_id is not None:
        message = f"<b><red>(record_id: {record_id})</></> - {message}"
    logger.opt(colors=True).log(level, message, *args)


def logger_default_config():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level><bold>{level: <8}</></> - <level>{message}</> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>",
        level="INFO",
        colorize=True,
    )


logger_default_config()
