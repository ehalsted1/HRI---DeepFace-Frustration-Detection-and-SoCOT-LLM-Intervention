
import logging
import os

LOG_FORMAT = "%(asctime)s [%(processName)s] %(levelname)s %(name)s: %(message)s"
_TRUTHY = {"1", "true", "yes", "on"}


def resolve_debug_flag(debug=None):
    """
    Resolve a debug flag that can be provided directly or read from the
    ROB514_DEBUG environment variable.
    """
    if debug is not None:
        return bool(debug)

    env_value = os.getenv("ROB514_DEBUG")
    if env_value is None:
        return False
    return env_value.strip().lower() in _TRUTHY


def init_logging(debug=None):
    """
    Initialise the root logger with a consistent format and level.
    Returns the resolved boolean debug flag that should be used by callers.
    """
    resolved_debug = resolve_debug_flag(debug)
    level = logging.DEBUG if resolved_debug else logging.INFO
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT)
    else:
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(LOG_FORMAT))

    return resolved_debug
