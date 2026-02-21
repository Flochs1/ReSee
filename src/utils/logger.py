"""Logging setup with colored console output."""

import logging
import sys
from typing import Optional
from colorama import Fore, Back, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels."""

    # Color mapping for different log levels
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
    }

    def __init__(self, fmt: str, use_colors: bool = True):
        """
        Initialize colored formatter.

        Args:
            fmt: Log format string.
            use_colors: Whether to use colors in output.
        """
        super().__init__(fmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.

        Args:
            record: Log record to format.

        Returns:
            Formatted log string with color codes.
        """
        if self.use_colors:
            # Get color for this log level
            color = self.COLORS.get(record.levelname, '')

            # Color the level name
            record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"

            # Color the message based on level
            if record.levelno >= logging.ERROR:
                record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
            elif record.levelno == logging.WARNING:
                record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"

        return super().format(record)


def setup_logger(
    name: str,
    level: str = "INFO",
    fmt: Optional[str] = None,
    use_colors: bool = True
) -> logging.Logger:
    """
    Set up a logger with colored console output.

    Args:
        name: Logger name (typically __name__).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt: Custom format string. If None, uses default format.
        use_colors: Whether to use colored output.

    Returns:
        Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Default format
    if fmt is None:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Create formatter
    if use_colors:
        formatter = ColoredFormatter(fmt, use_colors=True)
    else:
        formatter = logging.Formatter(fmt)

    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
