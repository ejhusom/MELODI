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

from config import config

def concatenate_data():

    # If main (concatenated data set does not already exists, create it.
    if not os.path.exists(config.MAIN_DATASET_PATH):  
        if config.SAVED_DATA_EXTENSION == ".csv":
            subprocess.run(["cat", f"{config.DATA_DIR_PATH}/*{config.LLM_DATA_FILENAME} > {config.MAIN_DATASET_PATH}"], check=True)
            print(f"Concatenated data into {config.MAIN_DATASET_PATH}.")
        else:
            raise Exception("Concatenating data not supported for other formats than .csv.")

def read_data():

    df = pd.read_csv(config.MAIN_DATASET_PATH)

    return df

def plot_data(df):

    df.plot()
    plt.show()

if __name__ == '__main__':
    concatenate_data()
    df = read_data()
    print(df)
    breakpoint()
    # plot_data(df)

