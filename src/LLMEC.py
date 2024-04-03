#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Use local LLMs while monitoring energy usage.

Author:
    Erik Johannes Husom

Created:
    2024-02-15

"""
import configparser
import datetime
import json
import os
import subprocess
import sys
import time

from _csv import Error as CSVErr

import ijson
import matplotlib.pyplot as plt
import pandas as pd

# from deepeval import assert_test, evaluate
# from deepeval.metrics import AnswerRelevancyMetric
# from deepeval.test_case import LLMTestCase

from config import config
from LLMAPIClient import LLMAPIClient

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
        prompt="How can we use Artificial Intelligence for a better society?",
        llm_service=None,
        llm_api_url=None,
        model_name=None,
        stream=False,
        save_power_data=False,
        plot_power_usage=False,
        task_type="unknown",
    ):
        """Prompts LLM and monitors energy consumption.

        Args:
            prompt (str or list of str): The prompt(s) to be sent to the LLM.
            llm_service (str, default=None): The LLM service to use.
            llm_api_url (str, default=None): The API URL of the LLM service.
            model_name (str, default=None): The model name for the request. Defaults to "mistral".
            stream (bool, default=False): Whether to stream the response. Defaults to False.
            save_power_data (bool, default=False): Save power usage data to file.
            plot_power_usage (bool, default=False): Plot power usage.
            TODO: batch_mode (bool, default=False): 
        """
        if llm_service is None:
            llm_service = self.config.get("General", "llm_service", fallback="ollama")

        if llm_api_url is None:
            llm_api_url = self.config.get("General", "llm_api_url", fallback="http://localhost:11434/api/chat")

        if model_name is None:
            model_name = self.config.get("General", "model_name", fallback="mistral")

        # LLM parameters
        llm_client = LLMAPIClient(
            llm_service=llm_service, api_url=llm_api_url, model_name=model_name, role="user"
        )

        if isinstance(prompt, str):
            prompts = [prompt]

        for p in prompts: 

            # Start power measurements
            if self.verbosity > 0:
                print("Starting power measurements...")

            metrics_process = subprocess.Popen(
                [
                    "scaphandre",
                    "json",
                    "--step", "0",
                    "--step-nano", str(config.SAMPLE_FREQUENCY_NANO_SECONDS),
                    "--resources",
                    "--process-regex", "ollama",
                    "--file", config.METRICS_STREAM_TEMP_FILE,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            time.sleep(config.MONITORING_START_DELAY)

            # Prompt LLM
            if self.verbosity > 0:
                print("Calling LLM service...")
            start_time = datetime.datetime.now()
            data = llm_client.call_api(prompt=p, stream=stream)
            end_time = datetime.datetime.now()

            if not data:
                print("Failed to get a response.")
                sys.exit(1)

            if self.verbosity > 0:
                print("Received response from LLM service.")

            # Collect power measurements
            time.sleep(config.MONITORING_END_DELAY)
            metrics_process.terminate()

            if self.verbosity > 0:
                print("Power measurements stopped.")

            with open( config.METRICS_STREAM_TEMP_FILE, "r") as f:
                metrics_stream = f.read()

            metrics = parse_json_objects(metrics_stream)
            if metrics == []:
                print("Found no metrics to parse.")
                continue
            metrics_per_process = self._parse_metrics(metrics)

            # print("==============================")
            # a = datetime.datetime.fromtimestamp(
            #         metrics_per_process["ollamaserve"]["timestamp"].iloc[0]
            # )
            # b = datetime.datetime.fromtimestamp(
            #         metrics_per_process["ollamaserve"]["timestamp"].iloc[-1]
            # )
            # print("Start time: ", start_time)
            # print("End time: ", start_time)
            # print("Duration: ", end_time - start_time)
            # print("scaph Start time: ", a)
            # print("scaph End time: ", b)
            # print("scaph Duration: ", b - a)
            # print("==============================")

            for cmdline, specific_process in metrics_per_process.items():
                specific_process.set_index("timestamp", inplace=True)
                if config.LLM_SERVICE_KEYWORD in cmdline:
                    metrics_llm = specific_process
                if config.MONITORING_SERVICE_KEYWORD in cmdline:
                    metrics_monitoring = specific_process

            if plot_power_usage:
                plot_metrics(metrics_llm, metrics_monitoring)

            energy_consumption_dict = calculate_energy_consumption_from_power_measurements(metrics_per_process)

            for cmdline, energy_consumption in energy_consumption_dict.items():
                # print(f"Energy consumption for cmdline "{cmdline[:10]}...": {energy_consumption} kWh")
                if config.LLM_SERVICE_KEYWORD in cmdline:
                    data["energy_consumption_llm"] = energy_consumption
                if config.MONITORING_SERVICE_KEYWORD in cmdline:
                    data["energy_consumption_monitoring"] = energy_consumption

            data["type"] = task_type

            data_df = pd.DataFrame.from_dict([data])

            if save_power_data:
                if self.verbosity > 0:
                    print("Saving data...")

                timestamp_filename = data["created_at"].replace(":", "").replace(".", "")
                llm_data_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.LLM_DATA_FILENAME}"
                metrics_llm_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.METRICS_LLM_FILENAME}"
                metrics_monitoring_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.METRICS_MONITORING_FILENAME}"

                with open(llm_data_filename, "w") as f:
                    self._save_data(data_df, llm_data_filename)

                with open(metrics_llm_filename, "w") as f:
                    self._save_data(metrics_llm, metrics_llm_filename)

                with open(metrics_monitoring_filename, "w") as f:
                    self._save_data(metrics_monitoring, metrics_monitoring_filename)

                if self.verbosity > 0:
                    print(f"Data saved with timestamp {timestamp_filename}")

        return data_df

    def _save_data(self, data, filename):

        with open(filename, "w") as f:
            if "pkl" in str(filename).split(os.extsep):
                data.to_pickle(filename)
            elif "csv" in str(filename).split(os.extsep):
                try:
                    data.to_csv(filename)
                except CSVErr as e:
                    print("Failed to save with escape character. Skipping save:", e)
                    # try:
                    #     print("Encountered CSV error:", e)
                    #     print("Retrying with escape character set...")
                    #     # If error, retry with escapechar
                    #     data.to_csv(filename, escapechar="\\")
                    # except CSVErr as e:
                    #     # If still an error, skip saving
                    #     print("Failed to save with escape character. Skipping save:", e)
            elif "json" in os.path.splitext(filename)[-1]:
                data.to_json(filename)
            else:
                print("Did not recognize file extension.")
                

    def _parse_metrics(self, metrics):
        """Convert collected power metrics into structured data."""

        dfs = []

        for measurement in metrics:
            consumers = measurement["consumers"]
            df = flatten_data(consumers, split_key="resources_usage")
            dfs.append(df)

        metrics_structured = pd.concat(dfs)
        metrics_per_process = split_dataframe_by_column(metrics_structured, "cmdline")

        return metrics_per_process


    def run_experiment(self,
                       dataset_path=None,
                       prompts=None,
                       task_type=None,
        ):

        if dataset_path:
            if dataset_path.endswith(".json"):
                raise ValueError("json format not yet supported.")
                # with open(dataset_path, "rb") as f:
                #     for record in ijson.items(f, "item"):
            elif dataset_path.endswith(".jsonl"):
                with open(dataset_path, "rb") as f:
                    for line in f.readlines():
                        obj = json.loads(line)
                        if "conversations" in obj:
                            conversation_label = "conversations"
                            role_label = "user"
                            human_label = "human"
                            content_label = "text"
                        elif "messages" in obj:
                            conversation_label = "messages"
                            role_label = "role"
                            human_label = "user"
                            content_label = "content"
                        else:
                            print("Could not locate prompt in dataset.")
                            sys.exit(1)

                        for conv in obj[conversation_label]:
                            if conv[role_label] == human_label:
                                prompt = conv[content_label]
                                df = self.run_prompt_with_energy_monitoring(
                                        prompt=prompt,
                                        save_power_data=True,
                                        plot_power_usage=False,
                                )
            elif dataset_path.endswith(".csv"):
                df = pd.read_csv(dataset_path)

                if task_type:
                    for prompt in df["prompt"]:
                        df = self.run_prompt_with_energy_monitoring(
                                prompt=prompt,
                                save_power_data=True,
                                plot_power_usage=False,
                                task_type=task_type,
                        )
                elif "type" in df.columns:
                    for index, row in df.iterrows():
                        df = self.run_prompt_with_energy_monitoring(
                                prompt=row["prompt"],
                                save_power_data=True,
                                plot_power_usage=False,
                                task_type=row["type"],
                        )
                else:
                    df = self.run_prompt_with_energy_monitoring(
                            prompt=prompt,
                            save_power_data=True,
                            plot_power_usage=False,
                            task_type="unknown"
                    )
            else:
                raise ValueError("Dataset must be in csv, json or jsonl format.")

            # Read dataset
        elif prompts:
            pass
            # Use prompts to run experiment
        else:
            raise ValueError("No dataset or prompts given. Cannot run experiment.")
                
def plot_metrics(metrics_llm, metrics_monitoring):
    """Plot metrics for a single prompt-response."""

    plt.figure()
    # plt.plot(
    #         metrics_monitoring.index,
    #         metrics_monitoring["consumption"],
    #         ".-",
    #         label="Monitoring service",
    # )
    plt.plot(
            metrics_llm.index,
            metrics_llm["consumption"],
            ".-",
            label="LLM service",
    )
    # metrics_monitoring["consumption"].plot(color="red")
    # metrics_llm["consumption"].plot()
    plt.xlabel("Timestamps")
    plt.ylabel("Power consumption (microwatts)")
    plt.legend()
    plt.show()

def parse_json_objects_from_file(file_path):
    """Wrapper for "parse_json_objects", to allow for passing file name."""

    # Open and read the content of the file
    with open(file_path, "r") as file:
        content = file.read()

    objects = parse_json_objects(content)

    return objects

def parse_json_objects(content):
    """Extract JSON objects from string.

    This function is specifically made for extracting JSON objects from a string
    that may contain multiple root objects, and also for allowing that the string
    may end with an incomplete JSON object (which are excluded from the
    parsing).

    Args:
        content (str): String to parse.

    Returns:
        objects (list): JSON objects.

    """

    objects = []  # List to store successfully parsed JSON objects
    start_idx = 0  # Start index of the JSON object being parsed

    # Iterate over the content to find JSON objects
    while True:
        try:
            obj, end_idx = json.JSONDecoder().raw_decode(content[start_idx:])
            objects.append(obj)
            start_idx += end_idx
        except json.JSONDecodeError:
            break  # Stop parsing as we"ve either reached the end or found incomplete JSON

    return objects


def flatten_data(data, split_key):
    """
    Flattens a list of dictionaries, including nested dictionaries,
    and converts it into a pandas DataFrame.

    Args:
        data (list): List of dictionaries to be flattened and converted.

    Returns:
        df (pdDataFrame): A pandas DataFrame with flattened data.
    """
    # Iterate over each record in the provided data list
    for record in data:
        # Check if there"s a nested dictionary that needs flattening
        # Here "resources_usage" is the nested dictionary we expect based on the given structure
        if split_key in record:
            # Extract and remove the nested dictionary
            split_key_items = record.pop(split_key)
            # Merge the nested dictionary"s key-value pairs into the main dictionary
            for key, value in split_key_items.items():
                record[key] = value

    # Convert the now-flattened list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    return df

def split_dataframe_by_column(df, column_name):
    """
    Splits a DataFrame into multiple DataFrames based on unique values in a specified column.

    Args:
        df: The pandas DataFrame to split.
        column_name: The name of the column to split the DataFrame by.

    Returns:
        A dictionary of DataFrames, where each key is a unique value from the
        specified column, and its corresponding value is a DataFrame containing
        only rows with that value.
    """
    # Initialize an empty dictionary to store the result
    df_dict = {}

    # Get unique values in the specified column
    unique_values = df[column_name].unique()

    # Loop through each unique value and create a separate DataFrame for it
    for value in unique_values:
        df_dict[value] = df[df[column_name] == value]

    return df_dict

def calculate_energy_consumption_from_power_measurements(df_dict):
    """
    Calculates the energy consumption in kWh for each DataFrame in the dictionary,
    given the power measurements in microwatts and using the timestamps to calculate
    the duration of each process.
    
    Parameters:
    - df_dict: Dictionary of DataFrames, split by unique cmdline values.
    
    Returns:
    - A dictionary with the same keys as df_dict, but the values are the total energy
      consumption in kWh for the processes corresponding to each cmdline, calculated
      using the duration derived from the timestamps.
    """
    energy_consumption_dict = {}

    for cmdline, df in df_dict.items():
        if not df.empty:
            # Convert timestamps to datetime objects
            df["datetime"] = pd.to_datetime(df.index, unit="s")
            # Calculate the duration in hours
            duration_hours = (df["datetime"].max() - df["datetime"].min()).total_seconds() / 3600.0

            # Handle the case where duration might be zero to avoid division by zero error
            if duration_hours > 0:
                # Calculate total power consumption in microwatts
                average_power_microwatts = df["consumption"].sum() / len(df)

                # Convert total power consumption to kWh
                energy_consumption_kwh = (average_power_microwatts * duration_hours) / 10**9

                # Store the result in the dictionary
                energy_consumption_dict[cmdline] = energy_consumption_kwh

            else:
                # If duration is zero, energy consumption is set to 0
                energy_consumption_dict[cmdline] = 0

    print(energy_consumption_dict["ollamaserve"])
    return energy_consumption_dict


if __name__ == "__main__":

    llm = LLMEC()
    # llm.run_experiment("/home/erikhu/Documents/datasets/Code-Feedback.jsonl")
    # llm.run_experiment("/home/erikhu/Documents/datasets/Code-Feedback-error.jsonl")
    # llm.run_experiment("/home/erikhu/Documents/datasets/test.jsonl")
    # llm.run_experiment("/home/erikhu/Documents/datasets/alpaca_prompts_only.csv")
    llm.run_experiment("/home/erikhu/Documents/datasets/alpaca_prompts_categorized_v1.csv")
    # llm.run_experiment("data/benchmark_datasets/sharegpt-english-small.jsonl")
    # llm.run_experiment("data/benchmark_datasets/sharegpt-english-very-small.jsonl")
    # llm.run_prompt_with_energy_monitoring(
    #     prompt="What is the capital in France?", save_power_data=True
    # )
