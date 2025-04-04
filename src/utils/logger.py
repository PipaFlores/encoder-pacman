import logging
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = None, level: str = "WARNING", log_to_file: bool = False
) -> logging.Logger:
    """
    Set up a logger with console handler and optional file handler.

    Args:
        name (str): Name of the logger (usually __name__ from the calling module)
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_to_file (bool): Whether to save logs to file (default: False)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name or __name__)

    # Prevent duplicate logging in Jupyter notebooks
    logger.propagate = False

    logger.setLevel(getattr(logging, level.upper()))

    # Only add handlers if they haven't been added before
    if not logger.handlers:
        # Create formatters
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")

        # File handler (optional)
        if log_to_file:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


# Default logger instance (console only)
logger = setup_logger("encoder_pacman")
