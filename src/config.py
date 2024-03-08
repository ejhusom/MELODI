#!/usr/bin/env python3
"""Global parameters for project.

Example:

    >>> from config import config
    >>> some_path = config.PARAMS_FILE_PATH
    >>> file = config.DATA_PATH / "filename.txt"

Author:   Erik Johannes Husom
Created:  2024-06-08

"""
from pathlib import Path

class Config:
    def __init__(self):
        # PARAMETERS

        # PATHS
        self.DATA_DIR_PATH = Path("./data/")
        self.CONFIG_FILE_PATH = Path("./config/config.ini")

        self._init_paths()

    def _init_paths(self):
        """Create directories if they don't exist."""
        directories = [
            self.DATA_DIR_PATH,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Instantiate a single configuration object to use throughout your application
config = Config()
