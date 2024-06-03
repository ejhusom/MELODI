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

        self.df["energy_per_token"] = self.df["energy_consumption_llm"] / self.df["response_token_length"]

        numeric_columns = [
            'total_duration', 'load_duration', 'prompt_token_length',
            'prompt_duration', 'response_token_length', 'response_duration',
            'energy_consumption_monitoring', 'energy_consumption_llm_cpu',
            'energy_consumption_llm_gpu', 'energy_consumption_llm_total',
            'energy_consumption_llm', "energy_per_token"
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
        print("Number of samples:", len(self.df))
        print("Mean energy consumption (kWh):", self.statistics.loc['energy_consumption_llm'].mean())
        print("Median energy consumption (kWh):", self.statistics.loc['energy_consumption_llm'].median())
        print("Standard deviation of energy consumption (kWh):", self.statistics.loc['energy_consumption_llm'].std())
        print("Mean energy consumption per token (kWh):", self.statistics.loc['energy_per_token'].mean())
        print("Median energy consumption per token (kWh):", self.statistics.loc['energy_per_token'].median())
        print("Standard deviation of energy consumption per token (kWh):", self.statistics.loc['energy_per_token'].std())


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
    def plot_single_correlations(df):
        # Calculate correlations with "energy_consumption_llm"
        correlations = df.corrwith(df["energy_consumption_llm"], numeric_only=True).sort_values(ascending=False)

        # Columns to drop
        drop_columns = [
            "energy_consumption_llm",
            "energy_consumption_llm_cpu",
            "energy_consumption_llm_gpu",
            "energy_consumption_llm_total",
            "energy_consumption_monitoring",
            "energy_per_token",
            "Unnamed: 0",
        ]

        for col in drop_columns:
            correlations.drop(col, inplace=True)

        # Only include the top 10 correlations
        correlations = correlations.head(10)

        # Create a bar plot for visualizing correlations
        plt.figure(figsize=(6, 4))
        sns.barplot(x=correlations.values, y=correlations.index, palette="viridis")
        # plt.title("Correlation with Total Energy Consumption")
        plt.xlabel("Correlation Coefficient")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR_PATH / "correlation_with_energy_consumption.pdf")
        plt.show()

    @staticmethod
    def compare_energy_per_token(datasets_dict):
        # Calculate energy consumption per generated token in the response

        # Colorblind friendly colors:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        # Extract promptset from dataset name
        promptsets = set([dataset_name.split("_")[0] for dataset_name in datasets_dict.keys()])
        # Make a dict matching promptset to color
        promptset_colors = {promptset: color for promptset, color in zip(promptsets, colors)}
        # Extract model names from dataset names
        model_names = set([dataset_name.split("_")[1] for dataset_name in datasets_dict.keys()])

        for dataset_name, dataset in datasets_dict.items():
            df = dataset["dataset"].df
            df['energy_per_token'] = df['energy_consumption_llm'] / df['response_token_length']

        fig, ax = plt.subplots(1, len(model_names), figsize=(5*len(model_names), 10))

        for i, model_name in enumerate(model_names):
            ax[i] = fig.add_subplot(1, len(model_names), i+1)
            datasets_for_model = [dataset["dataset"] for dataset in datasets_dict.values() if model_name in dataset["dataset"].name]
            dataset_names_for_model = [dataset.name for dataset in datasets_for_model]
            avg_energy_per_token_for_model = [dataset.df["energy_per_token"].mean() for dataset in datasets_for_model]

            current_model_data = pd.DataFrame({
                "dataset": dataset_names_for_model, 
                "energy_per_token": avg_energy_per_token_for_model,
                "promptset": [dataset_name.split("_")[0] for dataset_name in dataset_names_for_model],
                "model_size_and_hardware": ["_".join(dataset_name.split("_")[2:]) for dataset_name in dataset_names_for_model]
            })

            ax[i].bar(
                current_model_data["model_size_and_hardware"], 
                current_model_data["energy_per_token"], 
                color=current_model_data["promptset"].apply(lambda x: promptset_colors[x])
            )

            ax[i].set_xlabel(model_name)
            ax[i].set_ylabel('Energy consumption per token (kWh)')
            ax[i].set_xticks(ax[i].get_xticks(), rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def boxplot_comparison_subplots(datasets_dict, column="energy_consumption_llm", subplot_dimension="model_name", promptset_colors=None, filter_model_size=True):

        # Remove datasets with over 50b in model size
        if filter_model_size:
            datasets_dict = {dataset_name: dataset for dataset_name, dataset in datasets_dict.items() if int(dataset["dataset"].name.split("_")[2][:-1]) <= 50}

        if subplot_dimension == "promptset":
            category_set = set([dataset["dataset"].name.split("_")[0] for dataset in datasets_dict.values()])
        elif subplot_dimension == "model_name_and_size":
            category_set = set([dataset["dataset"].name.split("_")[1] + "_" + dataset["dataset"].name.split("_")[2] for dataset in datasets_dict.values()])
        elif subplot_dimension == "model_size":
            category_set = set([dataset["dataset"].name.split("_")[2] for dataset in datasets_dict.values()])
        elif subplot_dimension == "hardware":
            category_set = set([dataset["dataset"].name.split("_")[3].split(".")[0] for dataset in datasets_dict.values()])
        else: # model_name
            category_set = set([dataset["dataset"].name.split("_")[1] for dataset in datasets_dict.values()])

        # Sort category_set
        category_set = sorted(category_set)

        fig, ax = plt.subplots(1, len(category_set), figsize=(2.6*len(category_set), 4.5))

        for i, category in enumerate(category_set):
            datasets = [dataset["dataset"] for dataset in datasets_dict.values() if category in dataset["dataset"].name]
            dataset_colors = [dataset["color"] for dataset in datasets_dict.values() if category in dataset["dataset"].name]

            # Prepare the data for box plot
            data = [dataset.df[column] for dataset in datasets]
            dataset_names = [dataset.name for dataset in datasets]

            # Remove unnecessary parts of dataset names
            new_dataset_names = []
            for dataset_name in dataset_names:
                # Promptset is represented by color, so can be removed
                dataset_name = "_".join(dataset_name.split("_")[1:])
                # Remove category from dataset name
                dataset_name = dataset_name.replace(category, "")
                # Ensure there are no double underscores
                dataset_name = dataset_name.replace("__", "_")
                dataset_name = dataset_name.strip("_")
                new_dataset_names.append(dataset_name)

            # dataset_names = new_dataset_names

            bp = ax[i].boxplot(data, labels=dataset_names, patch_artist=True, showfliers=False, vert=1, notch=True)

            for j, (dataset, color) in enumerate(zip(datasets, dataset_colors)): 
                plt.setp(bp['boxes'][j], color=color)
                plt.setp(bp['medians'][j], color="red")

            # Add labels and title
            if i == 0:
                ax[i].set_ylabel(column)
            ax[i].set_xlabel(category)
            ax[i].set_xticklabels(new_dataset_names, rotation=45, ha='right')

        # Add legend to plot explaining which color corresponds to which promptset
        if promptset_colors:
            handles = [mpl.patches.Patch(color=color, label=promptset) for promptset, color in promptset_colors.items()]
            plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR_PATH / f"boxplot_comparison_subplots_{column}_{subplot_dimension}.pdf", bbox_inches='tight')
        plt.show()


    @staticmethod
    def boxplot_comparison(datasets_dict, column="energy_consumption_llm", promptset_colors=None, filter_model_size=True):

        # Remove datasets with over 50b in model size
        if filter_model_size:
            datasets_dict = {dataset_name: dataset for dataset_name, dataset in datasets_dict.items() if int(dataset["dataset"].name.split("_")[2][:-1]) <= 50}

        datasets = [dataset["dataset"] for dataset in datasets_dict.values()]

        # Prepare the data for box plot
        data = [dataset.df[column] for dataset in datasets]
        dataset_names = [dataset.name for dataset in datasets]
        # labels = [datasets_dict[dataset_name]["promptset"] for dataset_name in dataset_names]

        new_dataset_names = []
        for i, dataset_name in enumerate(dataset_names):
            dataset_name = "_".join(dataset_name.split("_")[1:])
            new_dataset_names.append(dataset_name)
        
        dataset_names = new_dataset_names

        hardware_setups = set([dataset["dataset"].name.split("_")[3].split(".")[0] for dataset in datasets_dict.values()])

        # Make different hatch patters for different hardware setups, up to 10 different patterns
        density = 4
        hatches = ["|", "-", "x", "\\", "o", "O", ".", "/", "*", "+"]
        hatches = [hatch * density for hatch in hatches]
        # Match different hatches to hardware setups
        hardware_hatches = {hardware: hatch for hardware, hatch in zip(hardware_setups, hatches)}

        # Plot the box plot
        fig, ax = plt.subplots(figsize=(5,6))
        bp = ax.boxplot(data, labels=dataset_names, patch_artist=True, showfliers=False, vert=1, notch=True)

        for i, dataset in enumerate(datasets_dict):
            plt.setp(bp['boxes'][i], color=datasets_dict[dataset]["color"])
            plt.setp(bp['medians'][i], color="red")

        for patch, hatch in zip(bp['boxes'], [hardware_hatches[dataset["dataset"].name.split("_")[3].split(".")[0]] for dataset in datasets_dict.values()]):
            patch.set_hatch(hatch)
            fc = patch.get_facecolor()
            patch.set_edgecolor(fc)
            patch.set_facecolor('white')


        # Add legend to plot explaining which hatch corresponds to which hardware setup
        handles = [mpl.patches.Patch(facecolor='white', edgecolor='black', hatch=hatch, label=hardware) for hardware, hatch in hardware_hatches.items()]
        legend_title = "Hardware setup"

        # Add legend to plot explaining which color corresponds to which promptset
        if promptset_colors:
            handles_promptset = [mpl.patches.Patch(color=color, label=promptset) for promptset, color in promptset_colors.items()]
            handles += handles_promptset
            legend_title += " / Promptset"

        # Add both handles to legend
        plt.legend(handles=handles, loc='upper left', title=legend_title)


        # Add labels and title
        ax.set_xlabel('Datasets')
        ax.set_ylabel(column)
        ax.set_title('Dataset comparison')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    datasets_path = config.DATA_DIR_PATH / "main_results"

    datasets = {}
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # loop through all files in the directory
    for filename in os.listdir(datasets_path):
        if filename.endswith('.csv'):  # check if the file is a CSV
            print(filename)
            file_path = os.path.join(datasets_path, filename)  # get full file path
            dataset = Dataset(file_path, name=filename.split(".")[0])
            # datasets[filename] = dataset
            datasets[filename] = {}
            datasets[filename]["dataset"] = dataset
            datasets[filename]["promptset"] = filename.split("_")[0]
            # datasets[filename]["model_size"] = int(filename.split("_")[2])[:-1])
            datasets[filename]["model_size"] = filename.split("_")[2]
            datasets[filename]["model_name"] = filename.split("_")[1]
            datasets[filename]["model_name_and_size"] = filename.split("_")[1] + "_" + filename.split("_")[2]
            datasets[filename]["hardware"] = filename.split("_")[3].split(".")[0]

    # Extract promptset from dataset names
    promptsets = set([dataset["promptset"] for dataset in datasets.values()])
    # Make a dict matching promptset to color
    promptset_colors = {promptset: color for promptset, color in zip(promptsets, colors)}
    # Extract model names from dataset names
    model_names = set([dataset["model_name"] for dataset in datasets.values()])
    # Extract hardware from dataset names
    hardware = set([dataset["hardware"] for dataset in datasets.values()])

    for dataset_name, dataset in datasets.items():
        dataset["color"] = promptset_colors[dataset["promptset"]]

    sorted_keys = sorted(
        datasets.keys(),
        key=lambda x: (x.split('_')[1], int(x.split('_')[2][:-1]))
    )
    datasets = {key: datasets[key] for key in sorted_keys}

    # Combine the dfs into one
    df = pd.concat([dataset["dataset"].df for dataset in datasets.values()])
    # df.to_csv(config.DATA_DIR_PATH / "all-results" / "combined_results.csv")
    # Plot correlation for all datasets
    # Dataset.plot_single_correlations(df)

    # Iterate through the datasets and print number of samples in each, mean energy consumption, etc.
    # for dataset_name, dataset in datasets.items():
    #     print(f"Dataset: {dataset_name}")
    #     dataset["dataset"].display_statistics()
    #     print("\n")


    # Dataset.compare_energy_per_token(datasets)

    Dataset.boxplot_comparison(datasets, column="energy_consumption_llm", promptset_colors=promptset_colors, filter_model_size=True)
    Dataset.boxplot_comparison(datasets, column="energy_per_token", promptset_colors=promptset_colors, filter_model_size=True)

    # Dataset.boxplot_comparison_subplots(datasets, column="energy_per_token",
    #                                     subplot_dimension="model_name_and_size",
    #                                     promptset_colors=promptset_colors,
    #                                     filter_model_size=False)
    # Dataset.boxplot_comparison_subplots(datasets, column="energy_consumption_llm",
    #                                     subplot_dimension="model_name_and_size",
    #                                     promptset_colors=promptset_colors,
    #                                     filter_model_size=False)
    # Dataset.boxplot_comparison_subplots(datasets, column="energy_per_token",
    #                                     subplot_dimension="hardware",
    #                                     promptset_colors=promptset_colors,
    #                                     filter_model_size=False)
    # Dataset.boxplot_comparison_subplots(datasets, column="energy_consumption_llm",
    #                                     subplot_dimension="hardware",
    #                                     promptset_colors=promptset_colors,
    #                                     filter_model_size=False)
