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

import ijson
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import tzlocal
from _csv import Error as CSVErr
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
from codecarbon import track_emissions
from energy_meter import EnergyMeter

from config import config
from LLMAPIClient import LLMAPIClient
from utils import joules2kwh

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

    def run_prompt_with_energy_monitoring_old(
        self,
        prompt="How can we use Artificial Intelligence for a better society?",
        llm_service=None,
        llm_api_url=None,
        model_name=None,
        save_power_data=False,
        plot_power_usage=False,
        task_type="unknown",
        monitoring_service="melodi",
    ):
        """Prompts LLM and monitors energy consumption.

        Args:
            prompt (str or list of str): The prompt(s) to be sent to the LLM.
            llm_service (str, default=None): The LLM service to use.
            llm_api_url (str, default=None): The API URL of the LLM service.
            model_name (str, default=None): The model name for the request. Defaults to "mistral".
            save_power_data (bool, default=False): Save power usage data to file.
            plot_power_usage (bool, default=False): Plot power usage.
            task_type (str, default="unknown"): The type of task the prompt
                asks for. This can be used to categorize the data.
            monitoring_service (str, default="melodi"): The monitoring service
                to use. Available services are "melodi", "pyjoules", "codecarbon",
                and "energymeter".
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

        # Make input prompt(s) iterable
        if isinstance(prompt, str):
            prompts = [prompt]

        failed_reading_data = False

        for p in prompts: 
            csv_handler = CSVHandler(config.PYJOULES_TEMP_FILE)

            em = EnergyMeter(disk_avg_speed=3000*1e6, # The average speed of your storage (see below how you can get it)
                  disk_active_power=0.1,    # How many Watts are used when the storage is reading or writing (you can usually find it in specs of your storage)
                  disk_idle_power=0.03,   # How many Watts are used when the storage is idle (you can usually find it in specs of your storage)
                  label=p,     # A label to identify the measurement, in this case the prompt
                  include_idle=False)     # If energy used during idle should be accounted for in the measurement. Defaults to False.

            @measure_energy(handler=csv_handler)
            @track_emissions(experiment_id=p, output_file=config.CODECARBON_TEMP_FILE)
            def run_inference_old():
                data = llm_client.call_api(prompt=p)
                return data

            # Start power measurements
            if self.verbosity > 0:
                print("Starting power measurements...")

            # Start nvidia-smi for monitoring GPU
            nvidiasmi_process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,power.draw",
                    "--format=csv",
                    "--loop-ms", str(config.SAMPLE_FREQUENCY_NANO_SECONDS/1e6), # Use the same frequency as scaphandre
                    "--filename", config.NVIDIASMI_STREAM_TEMP_FILE,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            scaphandre_process = subprocess.Popen(
                [
                    "scaphandre",
                    "json",
                    "--timeout", "10000000000",
                    "--step", "0",
                    "--step-nano", str(config.SAMPLE_FREQUENCY_NANO_SECONDS),
                    "--resources",
                    "--process-regex", "ollama",
                    "--file", config.SCAPHANDRE_STREAM_TEMP_FILE,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            time.sleep(config.MONITORING_START_DELAY)

            # Prompt LLM
            if self.verbosity > 0:
                print("Calling LLM service...")

            # Perform inference with LLM
            start_time = datetime.datetime.now(tz=pytz.utc)
            em.begin()
            data = run_inference_old()
            em.end()
            end_time = datetime.datetime.now(tz=pytz.utc)

            csv_handler.save_data()

            if not data:
                print("Failed to get a response.")
                sys.exit(1)

            if self.verbosity > 0:
                print("Received response from LLM service.")

            # Collect power measurements
            time.sleep(config.MONITORING_END_DELAY)
            scaphandre_process.terminate()
            nvidiasmi_process.terminate()

            if self.verbosity > 0:
                print("Power measurements stopped.")

            with open(config.SCAPHANDRE_STREAM_TEMP_FILE, "r") as f:
                metrics_stream = f.read()

            metrics = parse_json_objects(metrics_stream)
            if metrics == []:
                print("Found no metrics to parse.")
                continue

            metrics_per_process = self._parse_metrics(metrics)

            # Convert from microwatts to Watts
            for process in metrics_per_process:
                metrics_per_process[process]["consumption"] /= 1e6

            # Load GPU power draw measured by nvidia-smi:
            # Sometimes the writing of the GPU power data has not finished, so allow for some time if reading the data gives an errror.
            num_attempts = 5
            timeout = 3

            # Try reading the data for num_attempts times
            for i in range(num_attempts):
                try:
                    # Read nvidia-smi data, and skipping last row, since it sometimes is incomplete.
                    nvidiasmi_data = pd.read_csv(config.NVIDIASMI_STREAM_TEMP_FILE, on_bad_lines="skip")[:-1]
                    break  # Exit the loop if the read is successful
                except pd.errors.EmptyDataError:
                    if i < num_attempts - 1:
                        print(f"Error reading data. Waiting {timeout} second and trying again ({i+1}/{num_attempts})")
                        time.sleep(timeout)
                    else:
                        print(f"Error reading data after {num_attempts} attempts. Giving up.")
                        nvidiasmi_data = None  # Set the data to None if all attempts fail
                        # Continue with next prompt
                        failed_reading_data = True
                        break

            if failed_reading_data:
                print("Failed reading data. Skipping prompt.")
                continue

            try:
                nvidiasmi_data = self.postprocess_nvidiasmi_data(nvidiasmi_data)
            except:
                print("Failed postprocessing nvidiasmi data")
                continue

            # Read and postprocess PyJoules data, from a csv with ";" as separator
            # The columns are: timestamp, tag, duration, package_0, dram_0, core_0, uncore_0, and nvidia_gpu_0. Additional columns may exists depending on the hardware. The timestamp column has a UNIX timestamp, and should be converted to datetime. The columns wtih "gpu" in the name should be summed to get the total GPU power draw. The rest of the columns with energy measurements (package_*, dram_*, core_*, uncore_*) should be summed to get the total CPU power draw. The final df should consist of three columns: timestamp, duration, cpu_consumption, and gpu_consumption.
            try:
                pyjoules_data = pd.read_csv(config.PYJOULES_TEMP_FILE, sep=";")
                pyjoules_data["timestamp"] = pd.to_datetime(pyjoules_data["timestamp"], unit="s", utc=True)
                # Sum CPU consumption over all package/core/dram/uncore columns
                cpu_columns = [col for col in pyjoules_data.columns if "package" in col or "dram" in col or "core" in col or "uncore" in col]
                pyjoules_data["consumption"] = pyjoules_data[cpu_columns].sum(axis=1)
                # Sum GPU consumption over all gpu columns
                gpu_columns = [col for col in pyjoules_data.columns if "nvidia_gpu" in col]
                pyjoules_data["gpu_consumption"] = pyjoules_data[gpu_columns].sum(axis=1)
                # Drop the individual columns
                pyjoules_data = pyjoules_data[["timestamp", "duration", "consumption", "gpu_consumption"]]
                # Add column with total consumption
                pyjoules_data["total_consumption"] = pyjoules_data["consumption"] + pyjoules_data["gpu_consumption"]

                # The measurements are in microJoules, convert them to kWh
                pyjoules_data["consumption"] /= 3.6e12
                pyjoules_data["gpu_consumption"] /= 3.6e12
                pyjoules_data["total_consumption"] /= 3.6e12
            except Exception as e:
                print("Failed reading PyJoules data:", e)
                continue

            # Read and postprocess CodeCarbon data
            try:
                codecarbon_data = pd.read_csv(config.CODECARBON_TEMP_FILE)
                codecarbon_data = codecarbon_data.rename(columns={
                    "cpu_energy": "energy_consumption_llm_cpu_codecarbon",
                    "gpu_energy": "energy_consumption_llm_gpu_codecarbon",
                    "ram_energy": "energy_consumption_llm_ram_codecarbon",
                    "energy_consumed": "energy_consumption_llm_total_codecarbon"
                })
                codecarbon_data["energy_consumption_llm_cpu_codecarbon"] = (
                        codecarbon_data["energy_consumption_llm_cpu_codecarbon"] 
                        + codecarbon_data["energy_consumption_llm_ram_codecarbon"]
                )
            except:
                print("Failed reading CodeCarbon data.")
                continue


            # Save GPU power draw together with the other measurements
            metrics_per_process["llm_gpu"] = nvidiasmi_data

            print("==============================")
            a = datetime.datetime.fromtimestamp(
                    metrics_per_process["ollamaserve"]["timestamp"].iloc[0]
            )
            b = datetime.datetime.fromtimestamp(
                    metrics_per_process["ollamaserve"]["timestamp"].iloc[-1]
            )
            c = nvidiasmi_data["datetime"].iloc[0]
            d = nvidiasmi_data["datetime"].iloc[-1]
            print("Start time: ", start_time)
            print("End time: ", end_time)
            print("Duration: ", end_time - start_time)
            print("scaph Start time: ", a)
            print("scaph End time: ", b)
            print("scaph Duration: ", b - a)
            print("nvidia Start time: ", c)
            print("nvidia End time: ", d)
            print("nvidia Duration: ", d - c)
            print("==============================")

            for cmdline, specific_process in metrics_per_process.items():
                specific_process.set_index("timestamp", inplace=True)
                if config.LLM_SERVICE_KEYWORD in cmdline:
                    metrics_llm = specific_process
                if config.MONITORING_SERVICE_KEYWORD in cmdline:
                    metrics_monitoring = specific_process

            if plot_power_usage:
                plot_metrics(metrics_llm, metrics_monitoring, nvidiasmi_data)

            energy_consumption_dict = calculate_energy_consumption_from_power_measurements(metrics_per_process, start_time, end_time, show_plot=plot_power_usage)

            data["energy_consumption_llm_cpu"] = 0

            for cmdline, energy_consumption in energy_consumption_dict.items():
                if "gpu" in cmdline:
                    data["energy_consumption_llm_gpu"] = energy_consumption
                if config.LLM_SERVICE_KEYWORD in cmdline and not config.MONITORING_SERVICE_KEYWORD in cmdline:
                    data["energy_consumption_llm_cpu"] += energy_consumption
                if config.MONITORING_SERVICE_KEYWORD in cmdline:
                    data["energy_consumption_monitoring"] = energy_consumption

            data["type"] = task_type
            data["clock_duration"] = end_time - start_time
            # Convert datetime.timedelta to float number of seconds
            data["duration"] = data["clock_duration"].total_seconds()
            data["start_time"] = start_time
            data["end_time"] = end_time

            data_df = pd.DataFrame.from_dict([data])

            data_df["energy_consumption_llm_total"] = (
                    data_df["energy_consumption_llm_cpu"] +
                    data_df["energy_consumption_llm_gpu"]
            )

            data_df["energy_consumption_llm_cpu_codecarbon"] = codecarbon_data["energy_consumption_llm_cpu_codecarbon"]
            data_df["energy_consumption_llm_gpu_codecarbon"] = codecarbon_data["energy_consumption_llm_gpu_codecarbon"]
            data_df["energy_consumption_llm_total_codecarbon"] = codecarbon_data["energy_consumption_llm_total_codecarbon"]
            data_df["duration_codecarbon"] = codecarbon_data["duration"]

            data_df["energy_consumption_llm_cpu_pyjoules"] = pyjoules_data["consumption"].sum()
            data_df["energy_consumption_llm_gpu_pyjoules"] = pyjoules_data["gpu_consumption"].sum()
            data_df["energy_consumption_llm_total_pyjoules"] = pyjoules_data["total_consumption"].sum()
            data_df["duration_pyjoules"] = pyjoules_data["duration"]

            data_df["energy_consumption_llm_cpu_energymeter"] = em.get_total_joules_cpu()[0] + em.get_total_joules_dram()[0]
            data_df["energy_consumption_llm_gpu_energymeter"] = em.get_total_joules_gpu()
            data_df["energy_consumption_llm_total_energymeter"] = data_df["energy_consumption_llm_cpu_energymeter"] + data_df["energy_consumption_llm_gpu_energymeter"]

            data_df["energy_consumption_llm_cpu_energymeter"] = joules2kwh(data_df["energy_consumption_llm_cpu_energymeter"])
            data_df["energy_consumption_llm_gpu_energymeter"] = joules2kwh(data_df["energy_consumption_llm_gpu_energymeter"])
            data_df["energy_consumption_llm_total_energymeter"] = joules2kwh(data_df["energy_consumption_llm_total_energymeter"])

            if save_power_data:
                if self.verbosity > 0:
                    print("Saving data...")

                timestamp_filename = data["created_at"].replace(":", "").replace(".", "")
                llm_data_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.LLM_DATA_FILENAME}"
                metrics_llm_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.METRICS_LLM_FILENAME}"
                metrics_monitoring_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.METRICS_MONITORING_FILENAME}"
                metrics_llm_gpu_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.METRICS_LLM_GPU_FILENAME}"

                with open(llm_data_filename, "w") as f:
                    self._save_data(data_df, llm_data_filename)

                with open(metrics_llm_filename, "w") as f:
                    self._save_data(metrics_llm, metrics_llm_filename)

                with open(metrics_llm_gpu_filename, "w") as f:
                    self._save_data(nvidiasmi_data, metrics_llm_gpu_filename)

                with open(metrics_monitoring_filename, "w") as f:
                    self._save_data(metrics_monitoring, metrics_monitoring_filename)

                if self.verbosity > 0:
                    print(f"Data saved with timestamp {timestamp_filename}")

            # Delete temporary files
            config.remove_temp_files()

        try:
            return data_df
        except UnboundLocalError:
            print("Due to an error, no data was collected.")
            return None

    def run_prompt_with_energy_monitoring(
        self,
        prompt="How can we use Artificial Intelligence for a better society?",
        llm_service=None,
        llm_api_url=None,
        model_name=None,
        save_power_data=False,
        plot_power_usage=False,
        monitoring_service=None,
        task_type="unknown",
    ):
        """Prompts LLM and monitors energy consumption.

        Args:
            prompt (str or list of str): The prompt(s) to be sent to the LLM.
            llm_service (str, default=None): The LLM service to use.
            llm_api_url (str, default=None): The API URL of the LLM service.
            model_name (str, default=None): The model name for the request. Defaults to "mistral".
            save_power_data (bool, default=False): Save power usage data to file.
            plot_power_usage (bool, default=False): Plot power usage.
            task_type (str, default="unknown"): The type of task the prompt
                asks for. This can be used to categorize the data.
            monitoring_service (str, default=None): The monitoring service
                to use. Available services are "melodi", "pyjoules", "codecarbon",
                and "energymeter".
            TODO: batch_mode (bool, default=False): 
        """
        if llm_service is None:
            llm_service = self.config.get("General", "llm_service", fallback="ollama")

        if llm_api_url is None:
            llm_api_url = self.config.get("General", "llm_api_url", fallback="http://localhost:11434/api/chat")

        if model_name is None:
            model_name = self.config.get("General", "model_name", fallback="mistral")

        if monitoring_service is None:
            monitoring_service = self.config.get("General", "monitoring_service", fallback="melodi")

        if monitoring_service not in ["melodi", "pyjoules", "codecarbon", "energymeter"]:
            print("Monitoring service not recognized. Available services are 'melodi', 'pyjoules', 'codecarbon', and 'energymeter'.")
            sys.exit(1)

        # LLM parameters
        llm_client = LLMAPIClient(
            llm_service=llm_service, api_url=llm_api_url, model_name=model_name, role="user"
        )

        if self.verbosity > 0:
            print("Using LLM service: ", llm_service)
            print("Using monitoring service: ", monitoring_service)

        # Call the function that will measure the energy consumption
        # depending on which monitoring service is chosen.
        if monitoring_service == "melodi":
            data_df = self.run_prompt_with_energy_monitoring_melodi(
                llm_client,
                prompt=prompt,
                save_power_data=save_power_data,
                plot_power_usage=plot_power_usage,
                task_type=task_type,
            )
        elif monitoring_service == "pyjoules":
            data_df = self.run_prompt_with_energy_monitoring_pyjoules(
                llm_client,
                prompt=prompt,
                save_power_data=save_power_data,
                task_type=task_type,
            )
        elif monitoring_service == "codecarbon":
            data_df = self.run_prompt_with_energy_monitoring_codecarbon(
                llm_client,
                prompt=prompt,
                save_power_data=save_power_data,
                task_type=task_type,
            )
        elif monitoring_service == "energymeter":
            data_df = self.run_prompt_with_energy_monitoring_energymeter(
                llm_client,
                prompt=prompt,
                save_power_data=save_power_data,
                task_type=task_type,
            )
            
        return data_df

    def run_prompt_with_energy_monitoring_melodi(
        self,
        llm_client,
        prompt="How can we use Artificial Intelligence for a better society?",
        save_power_data=False,
        plot_power_usage=False,
        task_type="unknown",
    ):
        """Prompts LLM and monitors energy consumption.

        Args:
            prompt (str or list of str): The prompt(s) to be sent to the LLM.
            save_power_data (bool, default=False): Save power usage data to file.
            plot_power_usage (bool, default=False): Plot power usage.
            task_type (str, default="unknown"): The type of task the prompt
                asks for. This can be used to categorize the data.
            TODO: batch_mode (bool, default=False): 
        """

        failed_reading_data = False

        # Start power measurements
        if self.verbosity > 0:
            print("Starting power measurements...")

        # Start nvidia-smi for monitoring GPU
        nvidiasmi_process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,power.draw",
                "--format=csv",
                "--loop-ms", str(config.SAMPLE_FREQUENCY_NANO_SECONDS/1e6), # Use the same frequency as scaphandre
                "--filename", config.NVIDIASMI_STREAM_TEMP_FILE,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        scaphandre_process = subprocess.Popen(
            [
                "scaphandre",
                "json",
                "--timeout", "10000000000",
                "--step", "0",
                "--step-nano", str(config.SAMPLE_FREQUENCY_NANO_SECONDS),
                "--resources",
                "--process-regex", "ollama",
                "--file", config.SCAPHANDRE_STREAM_TEMP_FILE,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        time.sleep(config.MONITORING_START_DELAY)

        # Prompt LLM
        if self.verbosity > 0:
            print("Calling LLM service...")

        # Perform inference with LLM
        start_time = datetime.datetime.now(tz=pytz.utc)
        data = llm_client.call_api(prompt=prompt)
        end_time = datetime.datetime.now(tz=pytz.utc)

        if not data:
            print("Failed to get a response.")
            sys.exit(1)

        if self.verbosity > 0:
            print("Received response from LLM service.")

        # Collect power measurements
        time.sleep(config.MONITORING_END_DELAY)
        scaphandre_process.terminate()
        nvidiasmi_process.terminate()

        if self.verbosity > 0:
            print("Power measurements stopped.")

        # Read data from scaphandre
        with open(config.SCAPHANDRE_STREAM_TEMP_FILE, "r") as f:
            metrics_stream = f.read()

        metrics = parse_json_objects(metrics_stream)
        if metrics == []:
            print("Found no metrics to parse.")
            return None

        metrics_per_process = self._parse_metrics(metrics)

        # Convert from microwatts to Watts
        for process in metrics_per_process:
            metrics_per_process[process]["consumption"] /= 1e6

        # Load GPU power draw measured by nvidia-smi:
        nvidiasmi_data = self.read_nvidiasmi_data()

        if nvidiasmi_data is None:
            print("Failed reading nvidia-smi data. Skipping prompt.")
            return None

        try:
            nvidiasmi_data = self.postprocess_nvidiasmi_data(nvidiasmi_data)
        except:
            print("Failed postprocessing nvidiasmi data")
            return None

        # Save GPU power draw together with the other measurements
        metrics_per_process["llm_gpu"] = nvidiasmi_data

        print("==============================")
        a = datetime.datetime.fromtimestamp(
                metrics_per_process["ollamaserve"]["timestamp"].iloc[0]
        )
        b = datetime.datetime.fromtimestamp(
                metrics_per_process["ollamaserve"]["timestamp"].iloc[-1]
        )
        c = nvidiasmi_data["datetime"].iloc[0]
        d = nvidiasmi_data["datetime"].iloc[-1]

        if self.verbosity > 1:
            print("Start time: ", start_time)
            print("End time: ", end_time)
            print("Duration: ", end_time - start_time)
            print("scaph Start time: ", a)
            print("scaph End time: ", b)
            print("scaph Duration: ", b - a)
            print("nvidia Start time: ", c)
            print("nvidia End time: ", d)
            print("nvidia Duration: ", d - c)

        for cmdline, specific_process in metrics_per_process.items():
            specific_process.set_index("timestamp", inplace=True)
            if config.LLM_SERVICE_KEYWORD in cmdline:
                metrics_llm = specific_process
            if config.MONITORING_SERVICE_KEYWORD in cmdline:
                metrics_monitoring = specific_process

        if plot_power_usage:
            plot_metrics(metrics_llm, metrics_monitoring, nvidiasmi_data)

        energy_consumption_dict = calculate_energy_consumption_from_power_measurements(metrics_per_process, start_time, end_time, show_plot=plot_power_usage)

        # Initialize CPU consumption to 0, to enable adding all related
        # processes to it.
        data["energy_consumption_llm_cpu"] = 0

        for cmdline, energy_consumption in energy_consumption_dict.items():
            if "gpu" in cmdline:
                data["energy_consumption_llm_gpu"] = energy_consumption
            if config.LLM_SERVICE_KEYWORD in cmdline and not config.MONITORING_SERVICE_KEYWORD in cmdline:
                data["energy_consumption_llm_cpu"] += energy_consumption
            if config.MONITORING_SERVICE_KEYWORD in cmdline:
                data["energy_consumption_monitoring"] = energy_consumption

        data["type"] = task_type
        data["duration_clock"] = (end_time - start_time).total_seconds()
        data["duration_monitoring_service"] = data["duration_clock"]
        data["start_time"] = start_time
        data["end_time"] = end_time
        data["monitoring_service"] = "melodi"

        data_df = pd.DataFrame.from_dict([data])

        data_df["energy_consumption_llm_total"] = (
                data_df["energy_consumption_llm_cpu"] +
                data_df["energy_consumption_llm_gpu"]
        )

        if save_power_data:
            if self.verbosity > 0:
                print("Saving data...")

            timestamp_filename = data["created_at"].replace(":", "").replace(".", "")
            llm_data_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.LLM_DATA_FILENAME}"
            metrics_llm_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.METRICS_LLM_FILENAME}"
            metrics_monitoring_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.METRICS_MONITORING_FILENAME}"
            metrics_llm_gpu_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.METRICS_LLM_GPU_FILENAME}"

            with open(llm_data_filename, "w") as f:
                self._save_data(data_df, llm_data_filename)

            with open(metrics_llm_filename, "w") as f:
                self._save_data(metrics_llm, metrics_llm_filename)

            with open(metrics_llm_gpu_filename, "w") as f:
                self._save_data(nvidiasmi_data, metrics_llm_gpu_filename)

            with open(metrics_monitoring_filename, "w") as f:
                self._save_data(metrics_monitoring, metrics_monitoring_filename)

            if self.verbosity > 0:
                print(f"Data saved with timestamp {timestamp_filename}")

        # Delete temporary files
        config.remove_temp_files()

        try:
            return data_df
        except UnboundLocalError:
            print("Due to an error, no data was collected.")
            return None

    def run_prompt_with_energy_monitoring_pyjoules(
        self,
        llm_client,
        prompt="How can we use Artificial Intelligence for a better society?",
        save_power_data=False,
        task_type="unknown",
    ):
        """Prompts LLM and monitors energy consumption.

        Args:
            prompt (str or list of str): The prompt(s) to be sent to the LLM.
            save_power_data (bool, default=False): Save power usage data to file.
            task_type (str, default="unknown"): The type of task the prompt
                asks for. This can be used to categorize the data.
        """
        csv_handler = CSVHandler(config.PYJOULES_TEMP_FILE)

        @measure_energy(handler=csv_handler)
        def run_inference_pyjoules(llm_client, prompt):
            return llm_client.call_api(prompt=prompt)

        # Prompt LLM
        if self.verbosity > 0:
            print("Calling LLM service...")

        # Perform inference with LLM
        start_time = datetime.datetime.now(tz=pytz.utc)
        data = run_inference_pyjoules(llm_client, prompt)
        end_time = datetime.datetime.now(tz=pytz.utc)
        csv_handler.save_data()

        if not data:
            print("Failed to get a response.")
            sys.exit(1)

        if self.verbosity > 0:
            print("Received response from LLM service.")

        # Read and postprocess PyJoules data, from a csv with ";" as separator
        # The columns are: timestamp, tag, duration, package_0, dram_0, core_0, uncore_0, and nvidia_gpu_0. Additional columns may exists depending on the hardware. The timestamp column has a UNIX timestamp, and should be converted to datetime. The columns wtih "gpu" in the name should be summed to get the total GPU power draw. The rest of the columns with energy measurements (package_*, dram_*, core_*, uncore_*) should be summed to get the total CPU power draw. The final df should consist of three columns: timestamp, duration, cpu_consumption, and gpu_consumption.
        try:
            pyjoules_data = pd.read_csv(config.PYJOULES_TEMP_FILE, sep=";")
            pyjoules_data["timestamp"] = pd.to_datetime(pyjoules_data["timestamp"], unit="s", utc=True)
            # Sum CPU consumption over all package/core/dram/uncore columns
            cpu_columns = [col for col in pyjoules_data.columns if "package" in col or "dram" in col or "core" in col or "uncore" in col]
            pyjoules_data["consumption"] = pyjoules_data[cpu_columns].sum(axis=1)
            # Sum GPU consumption over all gpu columns
            gpu_columns = [col for col in pyjoules_data.columns if "nvidia_gpu" in col]
            pyjoules_data["gpu_consumption"] = pyjoules_data[gpu_columns].sum(axis=1)
            # Drop the individual columns
            pyjoules_data = pyjoules_data[["timestamp", "duration", "consumption", "gpu_consumption"]]
            # Add column with total consumption
            pyjoules_data["total_consumption"] = pyjoules_data["consumption"] + pyjoules_data["gpu_consumption"]

            # The measurements are in microJoules, convert them to kWh
            pyjoules_data["consumption"] /= 3.6e12
            pyjoules_data["gpu_consumption"] /= 3.6e12
            pyjoules_data["total_consumption"] /= 3.6e12
        except Exception as e:
            print("Failed reading PyJoules data:", e)
            return None

        data["type"] = task_type
        data["duration_clock"] = (end_time - start_time).total_seconds()
        data["start_time"] = start_time
        data["end_time"] = end_time
        data["monitoring_service"] = "pyjoules"

        data_df = pd.DataFrame.from_dict([data])

        data_df["energy_consumption_llm_cpu"] = pyjoules_data["consumption"].sum()
        data_df["energy_consumption_llm_gpu"] = pyjoules_data["gpu_consumption"].sum()
        data_df["energy_consumption_llm_total"] = pyjoules_data["total_consumption"].sum()
        data_df["duration_monitoring_service"] = pyjoules_data["duration"]

        if save_power_data:
            if self.verbosity > 0:
                print("Saving data...")

            timestamp_filename = data["created_at"].replace(":", "").replace(".", "")
            llm_data_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.LLM_DATA_FILENAME}"

            with open(llm_data_filename, "w") as f:
                self._save_data(data_df, llm_data_filename)

            if self.verbosity > 0:
                print(f"Data saved with timestamp {timestamp_filename}")

        # Delete temporary files
        config.remove_temp_files()

        try:
            return data_df
        except UnboundLocalError:
            print("Due to an error, no data was collected.")
            return None

    def run_prompt_with_energy_monitoring_codecarbon(
        self,
        llm_client,
        prompt="How can we use Artificial Intelligence for a better society?",
        save_power_data=False,
        task_type="unknown",
    ):
        """Prompts LLM and monitors energy consumption.

        Args:
            llm_client (LLMAPIClient): The LLM API client.
            prompt (str or list of str): The prompt(s) to be sent to the LLM.
            save_power_data (bool, default=False): Save power usage data to file.
            task_type (str, default="unknown"): The type of task the prompt
                asks for. This can be used to categorize the data.
        """
        @track_emissions(experiment_id=prompt, output_file=config.CODECARBON_TEMP_FILE)
        def run_inference_codecarbon(llm_client, prompt):
            return llm_client.call_api(prompt=prompt)

        # Prompt LLM
        if self.verbosity > 0:
            print("Calling LLM service...")

        # Perform inference with LLM
        start_time = datetime.datetime.now(tz=pytz.utc)
        data = run_inference_codecarbon(llm_client, prompt)
        end_time = datetime.datetime.now(tz=pytz.utc)

        if not data:
            print("Failed to get a response.")
            sys.exit(1)

        if self.verbosity > 0:
            print("Received response from LLM service.")

        # Read and postprocess CodeCarbon data
        try:
            codecarbon_data = pd.read_csv(config.CODECARBON_TEMP_FILE)
            codecarbon_data = codecarbon_data.rename(columns={
                "cpu_energy": "energy_consumption_llm_cpu_codecarbon",
                "gpu_energy": "energy_consumption_llm_gpu_codecarbon",
                "ram_energy": "energy_consumption_llm_ram_codecarbon",
                "energy_consumed": "energy_consumption_llm_total_codecarbon"
            })
            codecarbon_data["energy_consumption_llm_cpu_codecarbon"] = (
                    codecarbon_data["energy_consumption_llm_cpu_codecarbon"] 
                    + codecarbon_data["energy_consumption_llm_ram_codecarbon"]
            )
        except:
            print("Failed reading CodeCarbon data.")
            return None

        data["type"] = task_type
        data["duration_clock"] = (end_time - start_time).total_seconds()
        data["start_time"] = start_time
        data["end_time"] = end_time
        data["monitoring_service"] = "codecarbon"

        data_df = pd.DataFrame.from_dict([data])

        data_df["energy_consumption_llm_cpu"] = codecarbon_data["energy_consumption_llm_cpu_codecarbon"]
        data_df["energy_consumption_llm_gpu"] = codecarbon_data["energy_consumption_llm_gpu_codecarbon"]
        data_df["energy_consumption_llm_total"] = codecarbon_data["energy_consumption_llm_total_codecarbon"]
        data_df["duration_monitoring_service"] = codecarbon_data["duration"]

        if save_power_data:
            if self.verbosity > 0:
                print("Saving data...")

            timestamp_filename = data["created_at"].replace(":", "").replace(".", "")
            llm_data_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.LLM_DATA_FILENAME}"

            with open(llm_data_filename, "w") as f:
                self._save_data(data_df, llm_data_filename)

            if self.verbosity > 0:
                print(f"Data saved with timestamp {timestamp_filename}")

        # Delete temporary files
        config.remove_temp_files()

        try:
            return data_df
        except UnboundLocalError:
            print("Due to an error, no data was collected.")
            return None

    def run_prompt_with_energy_monitoring_energymeter(
        self,
        llm_client,
        prompt="How can we use Artificial Intelligence for a better society?",
        save_power_data=False,
        task_type="unknown",
    ):
        """Prompts LLM and monitors energy consumption.

        Args:
            llm_client (LLMAPIClient): The LLM API client.
            prompt (str or list of str): The prompt(s) to be sent to the LLM.
            save_power_data (bool, default=False): Save power usage data to file.
            task_type (str, default="unknown"): The type of task the prompt
                asks for. This can be used to categorize the data.
        """
        em = EnergyMeter(disk_avg_speed=3000*1e6, # The average speed of your storage (see below how you can get it)
              disk_active_power=0.1,    # How many Watts are used when the storage is reading or writing (you can usually find it in specs of your storage)
              disk_idle_power=0.03,   # How many Watts are used when the storage is idle (you can usually find it in specs of your storage)
              label=prompt,     # A label to identify the measurement, in this case the prompt
              include_idle=False)     # If energy used during idle should be accounted for in the measurement. Defaults to False.

        # Prompt LLM
        if self.verbosity > 0:
            print("Calling LLM service...")

        # Perform inference with LLM
        start_time = datetime.datetime.now(tz=pytz.utc)
        em.begin()
        data = llm_client.call_api(prompt=prompt)
        em.end()
        end_time = datetime.datetime.now(tz=pytz.utc)

        if not data:
            print("Failed to get a response.")
            sys.exit(1)

        if self.verbosity > 0:
            print("Received response from LLM service.")

        data["type"] = task_type
        data["duration_clock"] = (end_time - start_time).total_seconds()
        data["duration_monitoring_service"] = data["duration_clock"]
        data["start_time"] = start_time
        data["end_time"] = end_time
        data["monitoring_service"] = "energymeter"

        data_df = pd.DataFrame.from_dict([data])

        data_df["energy_consumption_llm_cpu"] = em.get_total_joules_cpu()[0] + em.get_total_joules_dram()[0]
        data_df["energy_consumption_llm_gpu"] = em.get_total_joules_gpu()
        data_df["energy_consumption_llm_total"] = data_df["energy_consumption_llm_cpu"] + data_df["energy_consumption_llm_gpu"]

        data_df["energy_consumption_llm_cpu"] = joules2kwh(data_df["energy_consumption_llm_cpu"])
        data_df["energy_consumption_llm_gpu"] = joules2kwh(data_df["energy_consumption_llm_gpu"])
        data_df["energy_consumption_llm_total"] = joules2kwh(data_df["energy_consumption_llm_total"])

        if save_power_data:
            if self.verbosity > 0:
                print("Saving data...")

            timestamp_filename = data["created_at"].replace(":", "").replace(".", "")
            llm_data_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.LLM_DATA_FILENAME}"

            with open(llm_data_filename, "w") as f:
                self._save_data(data_df, llm_data_filename)

            if self.verbosity > 0:
                print(f"Data saved with timestamp {timestamp_filename}")

        # Delete temporary files
        config.remove_temp_files()

        try:
            return data_df
        except UnboundLocalError:
            print("Due to an error, no data was collected.")
            return None

    def read_nvidiasmi_data(self):
        # Load GPU power draw measured by nvidia-smi:
        # Sometimes the writing of the GPU power data has not finished, so
        # allow for some time if reading the data gives an errror.
        num_attempts = 5
        timeout = 3

        # Try reading the data for num_attempts times
        for i in range(num_attempts):
            try:
                # Read nvidia-smi data, and skipping last row, since it
                # sometimes is incomplete.
                nvidiasmi_data = pd.read_csv(config.NVIDIASMI_STREAM_TEMP_FILE, on_bad_lines="skip")[:-1]
                return nvidiasmi_data
            except pd.errors.EmptyDataError:
                if i < num_attempts - 1:
                    print(f"Error reading data. Waiting {timeout} second and trying again ({i+1}/{num_attempts})")
                    time.sleep(timeout)
                else:
                    print(f"Error reading data after {num_attempts} attempts. Giving up.")
                    nvidiasmi_data = None  # Set the data to None if all attempts fail
                    return nvidiasmi_data

    def postprocess_nvidiasmi_data(self, df):
        """Postprocess nvidia-smi data.

        Args:
            df (pd.DataFrame): The data to postprocess.

        Returns:
            df (pd.DataFrame): The postprocessed data.

        """

        # Rename columns
        df = df.rename(columns={"timestamp": "datetime", " power.draw [W]": "consumption"})
        # Drop nan rows
        df = df.dropna()
        # Convert timestamp to datetime. Errors='coerce' will set NaT for invalid values.
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        # Drop rows where 'date_column' is NaT
        df = df.dropna(subset=['datetime'])
        # Sort values
        df = df.sort_values('datetime')
        # Detect the local time zone and convert the nvidia-smi timestamps to UTC
        local_tz = pytz.timezone(tzlocal.get_localzone_name())
        df['datetime'] = df['datetime'].dt.tz_localize(local_tz).dt.tz_convert('UTC')
        # Create column with unix timestamp
        df["timestamp"] = pd.to_datetime(df["datetime"]).astype(int) / 10**9
        # Convert measurements from string with unit to a float number
        df["consumption"] = df["consumption"].str.replace(r'\s*W', '', regex=True).astype(float)
        # Drop rows with negative timestamps (in the index)
        df = df[df.index >= 0]

        return df

    def _save_data(self, data, filename):

        with open(filename, "w") as f:
            if "pkl" in str(filename).split(os.extsep):
                data.to_pickle(filename)
            elif "csv" in str(filename).split(os.extsep):
                try:
                    data.to_csv(filename)
                except CSVErr as e:
                    print("Failed to save with escape character. Skipping save:", e)
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
            counter = 1
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
                                print(f"Prompt #{counter} of dataset {dataset_path}")
                                df = self.run_prompt_with_energy_monitoring(
                                        prompt=prompt,
                                        save_power_data=True,
                                        plot_power_usage=False,
                                )
                                counter += 1
            elif dataset_path.endswith(".csv"):
                df = pd.read_csv(dataset_path)

                if task_type:
                    for prompt in df["prompt"]:
                        print(f"Prompt #{counter} of dataset {dataset_path}")
                        df = self.run_prompt_with_energy_monitoring(
                                prompt=prompt,
                                save_power_data=True,
                                plot_power_usage=False,
                                task_type=task_type,
                        )
                        counter += 1
                elif "type" in df.columns:
                    for index, row in df.iterrows():
                        print(f"Prompt #{counter} of dataset {dataset_path}")
                        df = self.run_prompt_with_energy_monitoring(
                                prompt=row["prompt"],
                                save_power_data=True,
                                plot_power_usage=True,
                                task_type=row["type"],
                        )
                        counter += 1
                else:
                    for index, row in df.iterrows():
                        print(f"Prompt #{counter} of dataset {dataset_path}")
                        df = self.run_prompt_with_energy_monitoring(
                                prompt=row["prompt"],
                                save_power_data=True,
                                plot_power_usage=True,
                                task_type="unknown"
                        )
                        counter += 1
            else:
                raise ValueError("Dataset must be in csv, json or jsonl format.")

            # Read dataset
        elif prompts:
            pass
            # Use prompts to run experiment
        else:
            raise ValueError("No dataset or prompts given. Cannot run experiment.")
                
def plot_metrics(metrics_llm, metrics_monitoring, metrics_gpu):
    """Plot metrics for a single prompt-response."""

    plt.figure()
    plt.plot(
            metrics_monitoring.index,
            metrics_monitoring["consumption"],
            ".-",
            label="Monitoring service",
    )
    plt.plot(
            metrics_llm.index,
            metrics_llm["consumption"],
            ".-",
            label="LLM service (CPU)",
    )
    plt.plot(
            metrics_gpu.index,
            metrics_gpu["consumption"],
            ".-",
            label="LLM service (GPU)",
    )
    plt.xlabel("Timestamps")
    plt.ylabel("Power consumption (W)")
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

def calculate_energy_consumption_from_power_measurements(
        df_dict, 
        start_time,
        end_time,
        buffer_before=config.MEASUREMENTS_START_BUFFER,
        buffer_after=config.MEASUREMENTS_END_BUFFER,
        show_plot=False
    ):
    """
    Calculates the energy consumption in kWh for each DataFrame in the dictionary,
    given the power measurements in microwatts and using the timestamps to calculate
    the duration of each process.

    The function takes actual start and end time as input, since the recorded
    data may have buffer time periods at the beginning and end of recording.
    
    Parameters:
    - df_dict: Dictionary of DataFrames, split by unique cmdline values.
    - start_time: Actual start time of inference.
    - end_time: Actual end time of inference.
    
    Returns:
    - A dictionary with the same keys as df_dict, but the values are the total energy
      consumption in kWh for the processes corresponding to each cmdline, calculated
      using the duration derived from the timestamps.
    """
    energy_consumption_dict = {}

    old_dfs = []
    new_dfs = []

    for cmdline, df in df_dict.items():
        if not df.empty:
            # Convert timestamps to datetime objects
            df["datetime"] = pd.to_datetime(df.index, unit="s", utc=True)
            # df['timestamp_seconds'] = df['datetime'].dt.total_seconds()

            old_dfs.append(df.copy())

            # Apply buffer times before and after start_time and end_time
            start_time_with_buffer = start_time - pd.Timedelta(seconds=buffer_before)
            end_time_with_buffer = end_time + pd.Timedelta(seconds=buffer_after)

            # Filter rows based on start_time and end_time with buffer
            df = df[(df["datetime"] >= start_time_with_buffer) & (df["datetime"] <= end_time_with_buffer)]

            new_dfs.append(df.copy())

            # Calculate the duration
            duration = (df["datetime"].max() - df["datetime"].min()).total_seconds()

            # Handle the case where duration might be zero to avoid division by zero error
            if duration > 0:
                # # Calculate total power consumption in microwatts
                # average_power_watts = df["consumption"].sum() / len(df)

                # # Convert total power consumption to kWh
                # energy_consumption_kwh = (average_power_watts * duration) / (3600 * 10**3)

                #====================================================
                # Calculate the time interval between each data point
                # time_intervals = df["datetime"].diff().dt.total_seconds()
                # Calculate the energy consumption by integrating the power values over time
                # energy_consumption_kwh = (df["consumption"] * time_intervals).sum() / (10**3 * 3600)  # Convert Joules to kWh
                # print(f"kWh (time intervals): {energy_consumption_kwh}")

                energy_consumption_joules = np.trapz(df["consumption"], df.index)
                energy_consumption_kwh = joules2kwh(energy_consumption_joules)
                print(f"kWh (trapz)  : {energy_consumption_kwh} - ({cmdline})")

                # Store the result in the dictionary
                energy_consumption_dict[cmdline] = energy_consumption_kwh

            else:
                # If duration is zero, energy consumption is set to 0
                energy_consumption_dict[cmdline] = 0


    if show_plot:
        plot_metrics_truncated(old_dfs, new_dfs)

    return energy_consumption_dict

def plot_metrics_truncated(old_dfs, new_dfs):
    """Plot metrics for a single prompt-response."""

    # FIXME: Indexing with numbers does not work reliable, since the order of
    # the Dataframes may change. Need to identify them by the keywords in the
    # 'exe' column of each dataframe.
    old_metrics_llm = old_dfs[1]
    old_metrics_monitoring = old_dfs[0]
    old_metrics_gpu = old_dfs[2]
    metrics_llm = new_dfs[1]
    metrics_monitoring = new_dfs[0]
    metrics_gpu = new_dfs[2]

    plt.figure()
    plt.plot(
            old_metrics_monitoring.index,
            old_metrics_monitoring["consumption"],
            linewidth=5, alpha=0.5,
            label="Monitoring service",
    )
    plt.plot(
            old_metrics_llm.index,
            old_metrics_llm["consumption"],
            linewidth=5, alpha=0.5,
            label="LLM service (CPU)",
    )
    plt.plot(
            old_metrics_gpu.index,
            old_metrics_gpu["consumption"],
            linewidth=5, alpha=0.5,
            label="LLM service (GPU)",
    )
    plt.plot(
            metrics_monitoring.index,
            metrics_monitoring["consumption"],
            ".-",
            label="Monitoring service",
    )
    plt.plot(
            metrics_llm.index,
            metrics_llm["consumption"],
            ".-",
            label="LLM service (CPU)",
    )
    plt.plot(
            metrics_gpu.index,
            metrics_gpu["consumption"],
            ".-",
            label="LLM service (GPU)",
    )
    plt.xlabel("Timestamps")
    plt.ylabel("Power consumption (W)")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    filepath = sys.argv[1]
    llm = LLMEC()
    llm.run_experiment(filepath)

    # n = 10

    # for i in range(n):
    #     llm.run_prompt_with_energy_monitoring(
    #         prompt="What is the capital of the Marshall Islands?", save_power_data=True,
    #         # prompt="Explain the general theory of relativity", save_power_data=True,
    #         plot_power_usage=True,
    #     )

    # for i in range(1,26):
    #     print("==================================================")
    #     print(f"Running /home/erikhu/Documents/datasets/alpaca/alpaca_2300_5000_{str(i).zfill(2)}.csv")
         # llm.run_experiment(f"/home/erikhu/Documents/datasets/alpaca/alpaca_2300_5000_{str(i).zfill(2)}.csv")
