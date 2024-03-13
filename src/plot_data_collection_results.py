#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot results from data collection on energy usage of LLMs.

Author:
    Erik Johannes Husom

Created:
    2024-03-11

"""
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import config


def read_data(force_concatenation=False):

    # If main (concatenated data set does not already exists, create it.
    if not os.path.exists(config.MAIN_DATASET_PATH) or force_concatenation:
        if config.SAVED_DATA_EXTENSION == ".csv":

            dfs = []

            for filename in os.listdir(config.DATA_DIR_PATH):
                if filename.endswith(config.LLM_DATA_FILENAME):
                    try:
                        df = pd.read_csv(config.DATA_DIR_PATH / filename, header=0, index_col=0)
                        dfs.append(df)
                    except pd.errors.EmptyDataError:
                        print(f"{filename} is empty.")

            full_df = pd.concat(dfs)
            full_df.reset_index(inplace=True)
            full_df.to_csv(config.MAIN_DATASET_PATH)

            print(f"Concatenated data into {config.MAIN_DATASET_PATH}.")
        else:
            raise Exception("Concatenating data not supported for other formats than .csv.")
    else:
        print("Found existing concatenated data set.")
        full_df = pd.read_csv(config.MAIN_DATASET_PATH)

    return full_df

def plot_data(df):

    df.plot()
    plt.show()

    corr_columns = [
            "total_duration",
            "load_duration",
            "prompt_token_length",
            "prompt_duration",
            "response_token_length",
            "response_duration",
            "energy_consumption_monitoring",
            "energy_consumption_llm",
    ]
    corr_df = pd.DataFrame(df, columns=corr_columns)
    corr = corr_df.corr(
        method="pearson",
        numeric_only=True,
    ).round(2)
    sns.heatmap(corr, annot=True, vmin=-1.0, vmax=1.0, cmap="RdBu")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = read_data(force_concatenation=True)
    print(df.info())
    plot_data(df)
