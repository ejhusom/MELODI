#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Use local LLMs while monitoring energy usage.

Author:
    Erik Johannes Husom

Created:
    2024-06-03

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

from config import config

class Recalculate_LLMEC():

    def __init__(self):
        """Recalculate energy consumption for LLMs."""
        pass


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
                

    def recaculate_energy_consumption(self, dataset_path, hardware_to_adjust="gpu", power_adjustment_value=10):

        # Get a list of every csv file in dataset_path
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        csv_files = [f for f in csv_files if "metrics_monitoring" not in f]
        print("All files are found.")
        # Each file has a format of "YYYY-mm-ddTHHMMSSnnnnnnnnn_[type of data].csv"
        # There are 3 types of data: "llm_data", "metrics_llm", "metrics_llm_gpu", "metrics_monitoring". The last one is not interesting for us.
        # First we need to extract the timestamp from the filename, and then sort the files by timestamp.
        csv_files = sorted(csv_files, key=lambda x: x.split('_')[0])
        # Then we need to get a set of unique timestamps, since each timestamp has 4 files associated with it.
        timestamps = set([f.split('_')[0] for f in csv_files])
        # Sort the timestamps
        timestamps = sorted(timestamps)

        # Make a new directory to store the adjusted data
        adjusted_data_path = dataset_path + "adjusted_data"
        os.makedirs(adjusted_data_path, exist_ok=True)

        # Iterate through each timestamp
        for timestamp in timestamps:
            # Get the files associated with the timestamp
            files = [f for f in csv_files if timestamp in f]
            # Load the data from each file
            data = {}
            for file in files:
                data["_".join(file.split('_')[1:]).split(".")[0]] = pd.read_csv(dataset_path + file)
            # Adjust the power consumption and make a new df to store the adjusted data
            adjusted_data = data.copy()
            if hardware_to_adjust == "gpu":
                adjusted_data["metrics_llm_gpu"]["consumption"] = adjusted_data["metrics_llm_gpu"]["consumption"] - power_adjustment_value
            elif hardware_to_adjust == "cpu":
                adjusted_data["metrics_llm"]["consumption"] = adjusted_data["metrics_llm"]["consumption"] - power_adjustment_value
            else:
                print("Hardware to adjust not recognized.")
                return

            breakpoint()

            # Save the adjusted data



#         energy_consumption_dict = calculate_energy_consumption_from_power_measurements(metrics_per_process, start_time, end_time, show_plot=plot_power_usage)

#         for cmdline, energy_consumption in energy_consumption_dict.items():
#             # print(f"Energy consumption for cmdline "{cmdline[:10]}...": {energy_consumption} kWh")
#             if "gpu" in cmdline:
#                 data["energy_consumption_llm_gpu"] = energy_consumption
#             if config.LLM_SERVICE_KEYWORD in cmdline:
#                 data["energy_consumption_llm_cpu"] = energy_consumption
#             if config.MONITORING_SERVICE_KEYWORD in cmdline:
#                 data["energy_consumption_monitoring"] = energy_consumption

#         data["type"] = task_type
#         data["clock_duration"] = end_time - start_time
#         data["start_time"] = start_time
#         data["end_time"] = end_time

#         data_df = pd.DataFrame.from_dict([data])

#         data_df["energy_consumption_llm_total"] = (
#                 data_df["energy_consumption_llm_cpu"] +
#                 data_df["energy_consumption_llm_gpu"]
#         )

#         if save_power_data:
#             if self.verbosity > 0:
#                 print("Saving data...")

#             timestamp_filename = data["created_at"].replace(":", "").replace(".", "")
#             llm_data_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.LLM_DATA_FILENAME}"
#             metrics_llm_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.METRICS_LLM_FILENAME}"
#             metrics_monitoring_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.METRICS_MONITORING_FILENAME}"
#             metrics_llm_gpu_filename = config.DATA_DIR_PATH / f"{timestamp_filename}_{config.METRICS_LLM_GPU_FILENAME}"



#         with open(llm_data_filename, "w") as f:
#             self._save_data(data_df, llm_data_filename)

#         with open(metrics_llm_filename, "w") as f:
#             self._save_data(metrics_llm, metrics_llm_filename)

#         with open(metrics_llm_gpu_filename, "w") as f:
#             self._save_data(nvidiasmi_data, metrics_llm_gpu_filename)

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
                energy_consumption_joules = np.trapz(df["consumption"], df.index)
                energy_consumption_kwh = energy_consumption_joules / (10**3 * 3600)

                # Store the result in the dictionary
                energy_consumption_dict[cmdline] = energy_consumption_kwh

            else:
                # If duration is zero, energy consumption is set to 0
                energy_consumption_dict[cmdline] = 0

    return energy_consumption_dict

if __name__ == "__main__":

    filepath = sys.argv[1]
    llm = Recalculate_LLMEC()
    llm.recaculate_energy_consumption(filepath)
