#!/usr/bin/env python3
"""Global parameters for project.

Example:

    >>> from config import config
    >>> some_variable = config.PARAMETER_NAME
    >>> file = config.DATA_DIR_PATH / "filename.txt"

Author:   Erik Johannes Husom
Created:  2024-06-08

"""
from pathlib import Path

class Config:
    def __init__(self):
        # PARAMETERS
        self.SAMPLE_FREQUENCY_NANO_SECONDS = 100000000
        self.LLM_SERVICE_KEYWORD = "ollamaserve"
        self.MONITORING_SERVICE_KEYWORD = "scaphandre"
        self.MONITORING_START_DELAY = 1.0
        self.MONITORING_END_DELAY = 2.0
        self.MEASUREMENTS_START_BUFFER = 0.5
        self.MEASUREMENTS_END_BUFFER = 0.5
        self.OPENAI_API_COMPATIBLE_SERVICES = [
                "llamafile",
                "llama.cpp",
        ]

        # PATHS AND FILENAMES
        self.DATA_DIR_PATH = Path("./data/")
        self.PLOTS_DIR_PATH = Path("./plots/")
        self.MAIN_RESULTS_DIR_PATH = self.DATA_DIR_PATH / "main_results/"
        self.COMPLETE_RESULTS_DIR_PATH = self.DATA_DIR_PATH / "complete_results/"
        self.CONFIG_FILE_PATH = Path("./config/config.ini")
        self.SAVED_DATA_EXTENSION = ".csv"  # .json, .pkl.xz, .csv
        self.LLM_DATA_FILENAME = "llm_data" + self.SAVED_DATA_EXTENSION
        self.METRICS_LLM_FILENAME = "metrics_llm" + self.SAVED_DATA_EXTENSION
        self.METRICS_LLM_GPU_FILENAME = "metrics_llm_gpu" + self.SAVED_DATA_EXTENSION
        self.METRICS_MONITORING_FILENAME = "metrics_monitoring" + self.SAVED_DATA_EXTENSION
        self.MAIN_DATASET_PATH = self.DATA_DIR_PATH / "dataset.csv"
        self.MAIN_DATASET_WITH_FEATURES_PATH = self.DATA_DIR_PATH / "dataset_with_features.csv"
        self.COMPLETE_DATASET_PATH = self.COMPLETE_RESULTS_DIR_PATH / "complete_dataset.csv"
        self.SCAPHANDRE_STREAM_TEMP_FILE = "tmp_scaphandre.json"
        self.NVIDIASMI_STREAM_TEMP_FILE = "tmp_nvidiasmi.csv"

        self._init_paths()

    def _init_paths(self):
        """Create directories if they don't exist."""
        directories = [
            self.DATA_DIR_PATH,
            self.PLOTS_DIR_PATH,
            self.MAIN_RESULTS_DIR_PATH,
            self.COMPLETE_RESULTS_DIR_PATH,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Instantiate a single configuration object to use throughout your application
config = Config()
