#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze results.

Author:
    Erik Johannes Husom

Created:
    2024-05-28 tirsdag 13:57:24 

"""
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import config


class Dataset():

    def __init__(self, filename, name="dataset"):

        self.df = pd.read_csv(filename, index_col=0)
        self.name = name
        self.energy_consumption_column_name = "energy_consumption_llm"
        self.preprocess_data()
        self.statistics = self.calculate_statistics()

    def preprocess_data(self):
        # Convert timestamp columns to datetime format
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
        self.df['start_time'] = pd.to_datetime(self.df['start_time'], errors='coerce')
        self.df['end_time'] = pd.to_datetime(self.df['end_time'], errors='coerce')

        # Ensure other columns are in the correct format
        numeric_columns = [
            'total_duration', 'load_duration', 'prompt_token_length',
            'prompt_duration', 'response_token_length', 'response_duration',
            'energy_consumption_monitoring', 'energy_consumption_llm_cpu',
            'energy_consumption_llm_gpu', 'energy_consumption_llm_total',
            'energy_consumption_llm'
        ]

        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Handle any additional necessary preprocessing
        self.df.dropna(inplace=True)  # Drop rows with any NaN values

    def calculate_statistics(self):
        # Calculate descriptive statistics for relevant columns
        numeric_columns = [
            'total_duration', 'load_duration', 'prompt_token_length',
            'prompt_duration', 'response_token_length', 'response_duration',
            'energy_consumption_monitoring', 'energy_consumption_llm_cpu',
            'energy_consumption_llm_gpu', 'energy_consumption_llm_total',
            'energy_consumption_llm'
        ]

        statistics = self.df[numeric_columns].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
        statistics['median'] = self.df[numeric_columns].median()
        statistics['range'] = statistics['max'] - statistics['min']
        statistics['iqr'] = statistics['75%'] - statistics['25%']
        statistics['mode'] = self.df[numeric_columns].mode().iloc[0]
        statistics['skewness'] = self.df[numeric_columns].skew()
        statistics['kurtosis'] = self.df[numeric_columns].kurtosis()
        statistics['std_dev'] = self.df[numeric_columns].std()

        # Outlier detection using IQR method
        Q1 = self.df[numeric_columns].quantile(0.25)
        Q3 = self.df[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((self.df[numeric_columns] < (Q1 - 1.5 * IQR)) | (self.df[numeric_columns] > (Q3 + 1.5 * IQR))).sum()

        statistics['outliers'] = outliers

        return statistics

    def display_statistics(self):
        print("Descriptive Statistics:\n", self.statistics)

    def plot_distribution(self, include_density_plots=False):
        # Plot distribution of relevant columns
        numeric_columns = [
            'total_duration', 
            # 'load_duration', 
            'prompt_token_length',
            # 'prompt_duration', 
            'response_token_length', 
            # 'response_duration',
            # 'energy_consumption_monitoring', 
            'energy_consumption_llm_cpu',
            'energy_consumption_llm_gpu', 
            # 'energy_consumption_llm_total',
            'energy_consumption_llm'
        ]
        
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
        
    def perform_correlation_analysis(self):

        numeric_columns = [
            'total_duration', 
            # 'load_duration', 
            'prompt_token_length',
            # 'prompt_duration', 
            'response_token_length', 
            # 'response_duration',
            # 'energy_consumption_monitoring', 
            'energy_consumption_llm_cpu',
            'energy_consumption_llm_gpu', 
            # 'energy_consumption_llm_total',
            'energy_consumption_llm'
        ]

        # Calculate correlation matrix
        corr_matrix = self.df[numeric_columns].corr()

        # Plot correlation heatmap
        plt.figure(figsize=(10, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10}, vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_datasets(datasets):
        # FIXME: Plotting does not work yet

        stats = [
                "mean",
                "median",
                "std_dev",
                "min",
                "max",
                "range",
                "outliers",
        ]

        # # Define the data for the grouped bar chart
        # groups = ['alpaca_gemma_2b', 'alpaca_gemma_7b', 'alpaca_llama_8b']
        # numeric_columns = ['total_duration', 'load_duration', 'prompt_token_length', 'prompt_duration']
        # values = [[self.df[col].mean() for col in numeric_columns] for _ in groups]

        values = [[dataset.statistics[stat] for dataset in datasets] for stat in stats]
        dataset_names = [dataset.name for dataset in datasets]


        # Set the position of each group on the X-axis
        x = np.arange(len(datasets))

        # Set the width of the bars
        width = 0.2

        # Plot the grouped bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, stat in enumerate(stats):
            ax.bar(x + i * width, values[i], width, label=stat)

        # Add labels, title, and legend
        ax.set_xlabel('Datasets')
        ax.set_ylabel('Mean Value')
        ax.set_title('Comparison of Datasets')
        ax.set_xticks(x + width * len(numeric_columns) / 2)
        ax.set_xticklabels(dataset_names)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    alpaca_gemma_2b = Dataset(config.DATA_DIR_PATH / "final-results/alpaca-gemma2b/dataset.csv", name="alpaca-gemma-2b")
    # alpaca_gemma_2b.display_statistics()
    # alpaca_gemma_2b.plot_distribution()
    # alpaca_gemma_2b.perform_correlation_analysis()

    alpaca_gemma_7b = Dataset(config.DATA_DIR_PATH / "final-results/alpaca-gemma7b/dataset.csv", name="alpaca-gemma-7b")
    alpaca_llama_8b = Dataset(config.DATA_DIR_PATH / "final-results/alpaca-llama3-8b/dataset.csv", name="alpaca-llama3-8b")

    datasets = [
            alpaca_gemma_2b,
            alpaca_gemma_7b,
            alpaca_llama_8b
    ]

    Dataset.compare_datasets(datasets)

    # breakpoint()
