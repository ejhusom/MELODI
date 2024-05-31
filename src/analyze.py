#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze results.

Author:
    Erik Johannes Husom

Created:
    2024-05-28 tirsdag 13:57:24 

"""
import os
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
    def compare_datasets_plot(datasets_dict):
        stats = ["mean", "median", "std_dev", "min", "max", "range"]#, "outliers"]

        datasets = list(datasets_dict.values())

        # Gather statistics for each dataset
        values = [[dataset.statistics[stat]["energy_consumption_llm"] for stat in stats] for dataset in datasets]
        values = np.array(values).T  # Transpose the array to match the shape

        dataset_names = [dataset.name for dataset in datasets]

        # Set the position of each group on the X-axis
        x = np.arange(len(stats))

        # Set the width of the bars
        width = 0.2

        # Plot the grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, dataset_name in enumerate(dataset_names):
            ax.bar(x + i * width, values[:, i], width, label=dataset_name)

        # Add labels, title, and legend
        ax.set_xlabel('Statistics')
        ax.set_ylabel('Value')
        ax.set_title('Comparison of Datasets')
        ax.set_xticks(x + width * (len(dataset_names) - 1) / 2)
        ax.set_xticklabels(stats)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_boxplot(datasets_dict):

        # The datasets must be sorted first by model size, then by model name
        datasets_dict = dict(sorted(datasets_dict.items(), key=lambda x: (int(x[1].name.split("_")[2][:-1]), x[1].name.split("_")[1])))

        datasets = list(datasets_dict.values())

        # Remove datasets with over 50b in model size
        datasets = [dataset for dataset in datasets if int(dataset.name.split("_")[2][:-1]) <= 50]

        # Prepare the data for box plot
        data = [dataset.df["energy_consumption_llm"] for dataset in datasets]
        dataset_names = [dataset.name for dataset in datasets]

        # Plot the box plot
        fig, ax = plt.subplots(figsize=(4,5))
        bp = ax.boxplot(data, labels=dataset_names, patch_artist=True, showfliers=False, vert=1, notch=True)

        # Use colors depending on which dataset it is ("alpaca" or "codefeedback")
        # Colorblind friendly colors:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        new_dataset_names = []
        for i, dataset_name in enumerate(dataset_names):
            if "alpaca" in dataset_name:
                color = colors[0]
                color = colors[0]
            else:
                color = colors[1]
            # for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp['boxes'][i], color=color)
            plt.setp(bp['medians'][i], color="red")
            # Remove "alpaca" and "codefeedback" from the dataset name
            dataset_name = dataset_name.replace("alpaca_", "").replace("codefeedback_", "")
            new_dataset_names.append(dataset_name)

        # Set the x-axis labels to the dataset names
        ax.set_xticklabels(new_dataset_names)


        # Use another way to visualize the difference between the different models, using color or shape. Need to show the difference between "gemma", "llama3" and "codellama"

        
        # Add legend to plot explaining which color corresponds to which dataset
        alpaca_patch = mpl.patches.Patch(color=colors[0], label='Alpaca')
        codefeedback_patch = mpl.patches.Patch(color=colors[1], label='CodeFeedback')
        plt.legend(handles=[alpaca_patch, codefeedback_patch], loc='upper right')


        # Add labels and title
        ax.set_xlabel('Datasets')
        ax.set_ylabel('Energy consumption (kWh)')
        ax.set_title('Dataset comparison')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_datasets(datasets_dict):
        """Compare statistics of multiple datasets.

        Args:
            datasets_dict (dict): Dictionary containing Dataset objects as values.

        """
        stats = ["mean", "median", "std_dev", "min", "max", "range", "outliers"]

        datasets = list(datasets_dict.values())

        # Gather statistics for each dataset
        values = [[dataset.statistics[stat]["energy_consumption_llm"] for stat in stats] for dataset in datasets]
        values = np.array(values).T

        dataset_names = [dataset.name for dataset in datasets]

        # Print the statistics
        print(f"{'Statistic':<15} {' '.join([dataset_name for dataset_name in dataset_names])}")
        for i, stat in enumerate(stats):
            print(f"{stat:<15} {' '.join([str(value) for value in values[i]])}")

        # Print the statistics in a more readable format
        print("\nStatistics:")
        for i, stat in enumerate(stats):
            print(f"{stat.capitalize()}:")
            for j, dataset_name in enumerate(dataset_names):
                print(f"  {dataset_name}: {values[i][j]}")

        # Save the statistics in LaTeX format
        # The numbers should be shown in scientific notation, but with a fixed number of decimal places
        # The underscore character in the dataset names should be replaced with dashes
        with open("statistics.tex", "w") as f:
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{l" + " ".join(["r" for _ in dataset_names]) + "}\n")
            f.write("\\toprule\n")
            f.write("Statistic & ")
            for dataset_name in dataset_names:
                f.write(f"{dataset_name.replace('_', '-')} & ")
            f.write(" \\\\\n")
            f.write("\\midrule\n")
            for i, stat in enumerate(stats):
                f.write(f"{stat.capitalize()} & ")
                for j, dataset_name in enumerate(dataset_names):
                    f.write(f"{values[i][j]:.2e}")
                    if j < len(dataset_names) - 1:
                        f.write(" & ")
                f.write(" \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

    @staticmethod
    def comparison_plot(datasets_dict):
        """Compare datasets across several dimensions: Model type, model size, dataset, and hardware.

        Args:
            datasets_dict (dict): Dictionary containing Dataset objects as values.

        """
        pass

if __name__ == '__main__':

    datasets_path = config.DATA_DIR_PATH / "main_results"

    # alpaca_gemma_2b = Dataset(config.DATA_DIR_PATH / "main_results/alpaca_cpu_gemma2b.csv", name="alpaca-gemma-2b")

    # alpaca_gemma_2b = Dataset(config.DATA_DIR_PATH / "final-results/alpaca-gemma2b/dataset.csv", name="alpaca-gemma-2b")
    # alpaca_gemma_2b.display_statistics()
    # alpaca_gemma_2b.plot_distribution()
    # alpaca_gemma_2b.perform_correlation_analysis()

    # alpaca_gemma_7b = Dataset(config.DATA_DIR_PATH / "final-results/alpaca-gemma7b/dataset.csv", name="alpaca-gemma-7b")
    # alpaca_gemma_7b.display_statistics()
    # alpaca_gemma_7b.plot_distribution()
    # alpaca_gemma_7b.perform_correlation_analysis()

    # alpaca_llama_8b = Dataset(config.DATA_DIR_PATH / "final-results/alpaca-llama3-8b/dataset.csv", name="alpaca-llama3-8b")
    # alpaca_llama_8b.display_statistics()
    # alpaca_llama_8b.plot_distribution()
    # alpaca_llama_8b.perform_correlation_analysis()

    # datasets = [
    #         alpaca_gemma_2b,
    #         alpaca_gemma_7b,
    #         alpaca_llama_8b
    # ]

    # Dataset.compare_datasets(datasets)
    # Dataset.plot_boxplot(datasets)

    datasets = {}

    # loop through all files in the directory
    for filename in os.listdir(datasets_path):
        if filename.endswith('.csv'):  # check if the file is a CSV
            print(filename)
            file_path = os.path.join(datasets_path, filename)  # get full file path
            dataset = Dataset(file_path, name=filename.split(".")[0])
            datasets[filename] = dataset

    Dataset.plot_boxplot(datasets)
    # Dataset.compare_datasets(datasets)
