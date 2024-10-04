#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze results.

Author:
    Erik Johannes Husom

Created:
    2024-05-28

"""
import os

import matplotlib as mpl

mpl.rcParams['axes.formatter.useoffset'] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import config


class EnergyDataset():
    """Class for handling datasets.

    Attributes:
        filename (str): The filename of the dataset.
        df (pd.DataFrame): The dataset as a pandas DataFrame.
        metadata (dict): Metadata for the dataset.
        energy_consumption_column_name (str): The name of the column containing energy consumption.
        energy_consumption_llm_column_name (str): The name of the column containing energy consumption from LLM.
        numeric_columns (list): A list of column names containing numeric values.
        statistics (pd.DataFrame): A DataFrame containing statistics for the dataset.

    """

    def __init__(self, 
                 filename, 
                 dataset_name=None,
                 model_name=None,
                 model_size=None,
                 promptset=None,
                 hardware=None):
        """Initialize the EnergyDataset object.

        Args:
            filename (str): The filename of the dataset.
            dataset_name (str): The name of the dataset.
            model_name (str): The name of the model.
            model_size (str): The size of the model.
            promptset (str): The promptset used.
            hardware (str): The hardware used.

        """

        self.filename = filename
        self.df = pd.read_csv(filename, index_col=0)
        self.energy_consumption_column_name = "energy_consumption"
        self.energy_consumption_llm_column_name = "energy_consumption_llm"

        self.metadata = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "model_size": model_size,
            "promptset": promptset,
            "hardware": hardware
        }

        for key, value in self.metadata.items():
            if value is None:
                self.metadata[key] = self._infer_metadata(key)

        self.preprocess_data()
        self.statistics = self.calculate_statistics()

    def _infer_metadata(self, key):
        """Infer metadata.

        Args:
            key (str): The key to infer.

        Returns:
            str: The inferred metadata.

        """

        try:
            if key == "dataset_name":
                return os.path.basename(self.filename)
            elif key == "model_name":
                return self.df["model_name"].unique()[0]
            elif key == "model_size":
                # Try to find the model size in the model name, e.g. "gemma:2b".
                model_name = self.df["model_name"].unique()[0]
                if ":" in model_name:
                    return model_name.split(":")[1]
                else:
                    return "Unknown"
            elif key == "promptset":
                return self.df["promptset"].unique()[0]
            elif key == "hardware":
                return self.df["hardware"].unique()[0]
        except KeyError:
            print(f"Metadata ({key}) could not be inferred.")
            return "Unknown"

    def preprocess_data(self):
        """Preprocess the dataset.

        """
        # Convert timestamp columns to datetime format
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
        self.df['start_time'] = pd.to_datetime(self.df['start_time'], errors='coerce')
        self.df['end_time'] = pd.to_datetime(self.df['end_time'], errors='coerce')

        # Ensure other columns are in the correct format
        self.numeric_columns = [
            'total_duration', 
            'load_duration', 
            'prompt_token_length',
            'prompt_duration', 
            'response_token_length', 
            'response_duration',
        ]

        # Include every column starting with energy_consumption*
        self.numeric_columns += [col for col in self.df.columns if col.startswith(self.energy_consumption_column_name)]

        for col in self.numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Calculate energy consumption per token for all columns starting with energy_consumption, keeping the same suffix
        for col in [col for col in self.df.columns if col.startswith(self.energy_consumption_llm_column_name)]:
            new_col = f"{col}_per_token"
            self.df[new_col] = self.df[col] / self.df["response_token_length"]
            self.numeric_columns.append(new_col)

        # Handle any additional necessary preprocessing
        self.df.dropna(inplace=True)  # Drop rows with any NaN values

    def calculate_statistics(self):
        """Calculate statistics for the dataset.

        Returns:
            pd.DataFrame: A DataFrame containing the statistics.

        """
        statistics = self.df[self.numeric_columns].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
        statistics['median'] = self.df[self.numeric_columns].median()
        statistics['range'] = statistics['max'] - statistics['min']
        statistics['iqr'] = statistics['75%'] - statistics['25%']
        statistics['mode'] = self.df[self.numeric_columns].mode().iloc[0]
        statistics['skewness'] = self.df[self.numeric_columns].skew()
        statistics['kurtosis'] = self.df[self.numeric_columns].kurtosis()
        statistics['std_dev'] = self.df[self.numeric_columns].std()

        # Outlier detection using IQR method
        Q1 = self.df[self.numeric_columns].quantile(0.25)
        Q3 = self.df[self.numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((self.df[self.numeric_columns] < (Q1 - 1.5 * IQR)) | (self.df[self.numeric_columns] > (Q3 + 1.5 * IQR))).sum()

        statistics['outliers'] = outliers

        return statistics

    def display_statistics(self, latex=False):
        """Print statistics for the dataset, in this latex table format:

                & Prompt dataset & Model & Hardware & No. of prompts & Mean energy cons. & Mean response length \\
     1 & Alpaca & gemma:2b & Workstation & 123 & 0.1 & 10 \\

        """

        # Extract statistics
        statistics = self.statistics
        mean_energy_consumption = statistics.loc["energy_consumption_llm", "mean"]
        mean_response_length = statistics.loc["response_token_length", "mean"]
        number_of_prompts = len(self.df)

        # Print statistics. The energy consumption must be in scientific format with 2 decimals
        if latex:
            print(f"& {self.metadata['promptset']} & {self.metadata['model_name']} & {self.metadata['model_size']} & {self.metadata['hardware']} & {number_of_prompts} & {mean_energy_consumption:.2e} & {mean_response_length:.2f} \\\\")
        else:
            print(f"Dataset: {self.metadata['dataset_name']}")
            print(f"Prompt dataset: {promptset}")
            print(f"Model: {model_name}")
            print(f"Model size: {model_size}")
            print(f"Hardware: {hardware}")
            print(f"Number of prompts: {number_of_prompts}")
            print(f"Mean energy consumption: {mean_energy_consumption:.2e}")
            print(f"Mean response length: {mean_response_length:.2f}")

    def display_expanded_statistics(self):
        """Print statistics for the dataset."""

        # Extract statistics, where each column is a statistic and each row is a relevant column
        statistics = self.statistics

        relevant_columns = [
                "energy_consumption_llm",
                "energy_per_token",
                "response_token_length",
                "total_duration",
                "prompt_token_length"
        ]

        relevant_statistics = [
                "count",
                "mean",
                "std_dev",
                # "min",
                # "25%",
                "50%",
                # "75%",
                # "max",
                # "range",
                "median",
                "iqr",
        ]

        # Print statistics for each relevant column, values is printed in scientific format with 2 decimals
        for col in relevant_columns:
            print(f"Statistics for column: {col}")
            for stat in relevant_statistics:
                if stat in statistics.columns:
                    value = statistics.loc[col, stat]
                    if stat == "mean" or stat == "std_dev" or stat == "range":
                        print(f"{stat}: {value:.2e}")
                    else:
                        print(f"{stat}: {value}")

    def plot_distribution(self, columns=None, include_density_plots=False):
        """Plot the distribution of the dataset.

        Args:
            columns (list): A list of columns to plot.
            include_density_plots (bool): Whether to include density plots.

        """

        if columns is None:
            columns = self.numeric_columns
        
        # Set up the matplotlib figure
        if include_density_plots:
            fig, axes = plt.subplots(len(numeric_columns), 2, figsize=(15, 5*len(numeric_columns)))
            for i, col in enumerate(numeric_columns):
                # Histogram
                sns.histplot(self.df[col], bins=30, kde=False, ax=axes[i, 0])
                axes[i, 0].set_title(f'Histogram of {col}')
                axes[i, 0].set_xlabel(col)
                axes[i, 0].set_ylabel('Frequency')
                
                # Density Plot
                sns.kdeplot(self.df[col], ax=axes[i, 1], fill=True)
                axes[i, 1].set_title(f'Density Plot of {col}')
                axes[i, 1].set_xlabel(col)
                axes[i, 1].set_ylabel('Density')
        else:
            fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(10, 5*len(numeric_columns)))
            for i, col in enumerate(numeric_columns):
                # Histogram
                sns.histplot(self.df[col], bins=30, kde=False, ax=axes[i])
                # axes[i].set_title(f'Histogram of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        
    def perform_correlation_analysis(self, columns=None):

        if columns is None:
            columns = self.numeric_columns

        # Calculate correlation matrix
        corr_matrix = self.df[numeric_columns].corr()

        # Plot correlation heatmap
        plt.figure(figsize=(10, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10}, vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()
