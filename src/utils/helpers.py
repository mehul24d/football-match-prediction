"""
src/utils/helpers.py
--------------------
Shared utility functions used throughout the project.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML configuration file and return as a nested dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    logger.info(f"Configuration loaded from {config_path}")
    return cfg


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
    logger.debug(f"Random seed set to {seed}")


# ─── Path helpers ─────────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist; return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent.parent


# ─── Logging setup ───────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO", log_dir: str | Path | None = None) -> None:
    """Configure loguru logging, optionally writing to a file."""
    logger.remove()
    log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} – {message}"
    logger.add(
        sink=__import__("sys").stderr,
        level=log_level,
        format=log_format,
        colorize=True,
    )
    if log_dir:
        ensure_dir(log_dir)
        logger.add(
            sink=str(Path(log_dir) / "app_{time:YYYY-MM-DD}.log"),
            level=log_level,
            format=log_format,
            rotation="1 day",
            retention="30 days",
        )
