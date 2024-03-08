#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run experiments on LLMs and energy consumption.

Author:
    Erik Johannes Husom

Created:
    2024-02-15

"""
import configparser
import datetime
import json
import re
import subprocess
import sys

from config import config
from LLMAPIClient import LLMAPIClient
from PrometheusClient import PrometheusClient

def run_prompt_with_energy_monitoring_with_prometheus(
    model_name="mistral",
    prompt="How can we use Artificial Intelligence for a better society?",
    stream=False,
    save_data=False,
):
    """Demonstrates how to use the LLMAPIClient to send a request to the Ollama API.

    Args:
        model_name (str, optional): The model name for the request. Defaults to "mistral".
        prompt (str): The prompt to be sent.
        stream (bool, optional): Whether to stream the response. Defaults to False.
    """

    # LLM parameters
    llm_api_url = "http://localhost:11434/api/chat"
    ollama_client = LLMAPIClient(
        api_url=llm_api_url, model_name=model_name, role="user"
    )

    # Prometheus parameters
    prom_client = PrometheusClient()
    metric_name = 'scaph_process_power_consumption_microwatts{cmdline=~".*ollama.*"}'

    # Prompt LLM
    data = ollama_client.call_api(prompt=prompt, stream=stream)
    end_time = datetime.datetime.now()
    if data:
        print(data)
    else:
        print("Failed to get a response.")
        sys.exit(1)

    metric_data = prom.get_metric_range_data(
        metric_name=metric_name,
        start_time=data["created_at"],
        end_time=end_time,
    )

    if save_data:
        timestamp_filename = data["created_at"].replace(":", "").replace(".", "")
        with open(config.DATA_DIR_PATH + f"llm_response_{timestamp_filename}.json", "w") as f:
            json.dump(data, f)

class LLMEC():

    def __init__(self, config_path=config.CONFIG_FILE_PATH):
        """Prompting LLMs while measuring the energy consumption.

        Args:
            config_path (str): The path to the configuration file for the
                explanation generator.

        """

        self.config = self._read_config(config_path)
        self.verbosity = int(self.config.get("General", "verbosity", fallback=0))

    def _read_config(self, config_path):
        """
        Reads the configuration settings from the specified file.

        Args:
            config_path (str): The path to the configuration file.

        """
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def run_prompt_with_energy_monitoring(
        self,
        model_name=None,
        prompt="How can we use Artificial Intelligence for a better society?",
        stream=False,
        save_data=False,
    ):
        """Demonstrates how to use the LLMAPIClient to send a request to the Ollama API.

        Args:
            model_name (str, optional): The model name for the request. Defaults to "mistral".
            prompt (str): The prompt to be sent.
            stream (bool, optional): Whether to stream the response. Defaults to False.
        """

        if model_name is None:
            model_name = self.config.get("General", "model_name", fallback="mistral")

        # LLM parameters
        llm_api_url = "http://localhost:11434/api/chat"
        ollama_client = LLMAPIClient(
            api_url=llm_api_url, model_name=model_name, role="user"
        )

        # Start power measurements
        # metrics_filename = config.DATA_DIR_PATH + "metrics.json" 
        if self.verbosity > 0:
            print("Starting power measurements...")
        metrics_process = subprocess.Popen(
            [
                "scaphandre",
                "json",
                "--step", "0",
                "--step-nano", "100000000",
                "--process-regex", "ollama",
                # "--file", metrics_filename,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Prompt LLM
        if self.verbosity > 0:
            print("Calling LLM API...")
        data = ollama_client.call_api(prompt=prompt, stream=stream)
        end_time = datetime.datetime.now()

        if not data:
            print("Failed to get a response.")
            sys.exit(1)

        if self.verbosity > 0:
            print("Received response from LLM API.")

        # Collect power measurements
        metrics_process.terminate()
        if self.verbosity > 0:
            print("Power measurements stopped.")
        metrics_stream = metrics_process.stdout.readlines()[-1].decode("utf-8")
        # with open(metrics_filename, "r") as f:
        #     metrics = json.load(f)
        metrics = parse_json_objects(metrics_stream)

        if save_data:
            if self.verbosity > 0:
                print("Saving data...")

            timestamp_filename = data["created_at"].replace(":", "").replace(".", "")
            filename = config.DATA_DIR_PATH / f"llm_response_{timestamp_filename}.json"
            with open(filename, "w") as f:
                json.dump(data, f)

            if self.verbosity > 0:
                print(f"Data saved to {filename}")

def parse_json_objects_from_file(file_path):
    # Open and read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    objects = parse_json_objects(content)

    return objects

def parse_json_objects(content):

    objects = []  # List to store successfully parsed JSON objects
    start_idx = 0  # Start index of the JSON object being parsed

    # Iterate over the content to find JSON objects
    while True:
        try:
            obj, end_idx = json.JSONDecoder().raw_decode(content[start_idx:])
            objects.append(obj)
            start_idx += end_idx
        except json.JSONDecodeError:
            break  # Stop parsing as we've either reached the end or found incomplete JSON

    return objects

if __name__ == "__main__":

    llm = LLMEC()
    llm.run_prompt_with_energy_monitoring(
        prompt="What is the capital in France?", save_data=True
    )
