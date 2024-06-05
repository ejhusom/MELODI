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
        """Print statistics for the dataset, in this latex table format:

                & Prompt dataset & Model & Hardware & No. of prompts & Mean energy cons. & Mean response length \\
     1 & Alpaca & gemma:2b & Workstation & 123 & 0.1 & 10 \\

        """

        # Extract model name and size from dataset name
        promptset = self.name.split("_")[0]
        model_name = self.name.split("_")[1]
        model_size = self.name.split("_")[2]
        hardware = self.name.split("_")[3].split(".")[0]

        # Extract statistics
        statistics = self.statistics

        # Extract relevant statistics
        mean_energy_consumption = statistics.loc["energy_consumption_llm", "mean"]
        mean_response_length = statistics.loc["response_token_length", "mean"]
        number_of_prompts = len(self.df)

        # Print statistics. The energy consumption must be in scientific format with 2 decimals
        print(f"& {promptset} & {model_name} & {model_size} & {hardware} & {number_of_prompts} & {mean_energy_consumption:.2e} & {mean_response_length:.2f} \\\\")

    def display_expanded_statistics(self):
        """Print statistics for the dataset."""

        # Extract model name and size from dataset name
        promptset = self.name.split("_")[0]
        model_name = self.name.split("_")[1]
        model_size = self.name.split("_")[2]
        hardware = self.name.split("_")[3].split(".")[0]

        # Extract statistics, where each column is a statistic and each row is a relevant column
        statistics = self.statistics

        relevant_columns = [
                "energy_consumption_llm",
                "energy_per_token",
                "response_token_length",
                # "total_duration",
                # "prompt_token_length"
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

        # Print statistics. The energy consumption must be in scientific format with 2 decimals

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
    def plot_single_correlations(df, num_corr=25):
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
            "Unnamed: 0.1",
            "index"
        ]

        for col in drop_columns:
            try:
                correlations.drop(col, inplace=True)
            except KeyError:
                pass

        # Only include the top num_corr correlations
        correlations = correlations.head(num_corr)

        # Split DataFrame into two parts
        df = pd.DataFrame(correlations)
        mid_index = len(df) // 2 + len(df) % 2
        df1 = df.iloc[:mid_index]
        df2 = df.iloc[mid_index:]
        # Move the index to a column and c  reate a numeric index for the two parts
        df1.reset_index(inplace=True)
        df2.reset_index(inplace=True)

        # Concatenate df1 and df2 into a single DataFrame, with df1 on the left and df2 on the right
        df = pd.concat([df1, df2], axis=1)
        df.columns = ["Feature", "Correlation", "Feature", "Correlation"]

        # Make a latex table and save to variable. Round the correlation values to 2 decimals
        latex_table = df.to_latex(index=False, float_format="%.3f")
        # Ensure that every underscore is escaped
        latex_table = latex_table.replace("_", "\_")
        # Print the latex table
        print(latex_table)


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

        # Ensure that the subplots needing less space are narrower
        # Loop through the category set and count the number of box needed in each box plot
        n_boxes = [len([dataset for dataset in datasets_dict.values() if category in dataset["dataset"].name]) for category in category_set]

        # fig, ax = plt.subplots(1, len(category_set), figsize=(2.0*len(category_set), 3.5))
        fig, ax = plt.subplots(1, len(category_set), figsize=(sum(n_boxes)/2+0.5, 3.2), gridspec_kw={'width_ratios': n_boxes})

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
                ax[i].set_ylabel(column + " (kWh)")
            ax[i].set_xlabel(category)
            ax[i].set_xticklabels(new_dataset_names, rotation=45, ha='right')

        # Add legend to plot explaining which color corresponds to which promptset
        # if promptset_colors:
        #     handles = [mpl.patches.Patch(color=color, label=promptset) for promptset, color in promptset_colors.items()]
        #     bb = (fig.subplotpars.left+0.1, fig.subplotpars.top+0.005)
        #           # fig.subplotpars.right*0.4-fig.subplotpars.,.05)

        #     ax[0].legend(handles=handles, bbox_to_anchor=bb, mode="expand", loc="lower left", title="Prompt dataset",
        #                    ncol=2, borderaxespad=0., bbox_transform=fig.transFigure, framealpha=0, edgecolor="white")

        plt.tight_layout()
        if filter_model_size:
            plot_filename = f"boxplot_comparison_subplots_{column}_{subplot_dimension}_large_models_removed"
        else:
            plot_filename = f"boxplot_comparison_subplots_{column}_{subplot_dimension}"
        plt.savefig(config.PLOTS_DIR_PATH / f"{plot_filename}.pdf", bbox_inches='tight')
        # plt.show()


    @staticmethod
    def boxplot_comparison(datasets_dict, column="energy_consumption_llm",
                           promptset_colors=None, filter_model_size=True,
                           shade_by_model_name_only=False, showlegend=True, hardware_hatches=None):

        # Remove datasets with over 50b in model size
        if filter_model_size:
            datasets_dict = {dataset_name: dataset for dataset_name, dataset in datasets_dict.items() if int(dataset["dataset"].name.split("_")[2][:-1]) <= 50}

        datasets = [dataset["dataset"] for dataset in datasets_dict.values()]

        # If column is energy_consumption_llm, rename this column "energy_per_response"
        if column == "energy_consumption_llm":
            new_column_name = "energy_per_response"
            for dataset in datasets:
                dataset.df[new_column_name] = dataset.df["energy_consumption_llm"]
            column = new_column_name

        # Prepare the data for box plot
        data = [dataset.df[column] for dataset in datasets]
        dataset_names = [dataset.name for dataset in datasets]

        new_dataset_names = []
        # Remove unnecessary parts of dataset names
        for i, dataset_name in enumerate(dataset_names):
            # Remove promptset 
            dataset_name = "_".join(dataset_name.split("_")[1:])
            if not filter_model_size and int(dataset_name.split("_")[1][:-1]) < 50:
                new_dataset_names.append(dataset_name)
            else:
                # Remove hardware
                dataset_name = "_".join(dataset_name.split("_")[:-1])
                new_dataset_names.append(dataset_name)
        
        dataset_names = new_dataset_names

        if filter_model_size:
            if showlegend:
                figsize = (5.1, 4.5)
            else:
                figsize = (4, 4)
        else:
            if showlegend:
                figsize = (4.4, 4.0)
            else:
                figsize = (3, 3.5)

        # Plot the box plot
        fig, ax = plt.subplots(figsize=figsize)
        bp = ax.boxplot(data, labels=dataset_names, patch_artist=True,
                        showfliers=False, vert=1, notch=True,
                        boxprops=dict(linewidth=0.5),
                        whiskerprops=dict(linewidth=0.5),
                        medianprops=dict(linewidth=0.5),
                        meanprops=dict(linewidth=0.5),
                        capprops=dict(linewidth=0.5))

        for i, dataset in enumerate(datasets_dict):
            plt.setp(bp['boxes'][i], color=datasets_dict[dataset]["color"])
            plt.setp(bp['medians'][i], color="red")

        for patch, hatch in zip(bp['boxes'], [hardware_hatches[dataset["dataset"].name.split("_")[3].split(".")[0]] for dataset in datasets_dict.values()]):
            patch.set_hatch(hatch)
            patch.set_linewidth(0.5)
            fc = patch.get_facecolor()
            patch.set_edgecolor(fc)
            patch.set_facecolor('white')

        # Add legend to plot explaining which hatch corresponds to which hardware setup
        handles = [mpl.patches.Patch(facecolor='white', edgecolor='black', hatch=hatch, label=hardware) for hardware, hatch in hardware_hatches.items()]
        legend_title = "Hardware setup"

        # Identify unique model names and their positions
        if shade_by_model_name_only:
            model_names = sorted(set([name.split("_")[0] for name in dataset_names]))
        else:
            # Sort model names ([modelname]_[size]b) alphabetically by the model name and the numerically by the size
            model_names = sorted(
                set([name for name in dataset_names]),
                key=lambda x: (x.split("_")[0], int(x.split("_")[1][:-1]))
            )
        model_positions = {model: [] for model in model_names}
        for i, name in enumerate(dataset_names):
            if shade_by_model_name_only:
                model_name = name.split("_")[0]
            else:
                model_name = name
            model_positions[model_name].append(i + 1)

        # Shade background based on model names
        for i, model in enumerate(model_names):
            positions = model_positions[model]
            start = positions[0] - 0.5
            end = positions[-1] + 0.5
            color = 'grey' if i % 2 == 0 else 'whitesmoke'
            ax.axvspan(start, end, color=color, alpha=0.3)

            # Add label for each shaded area
            if shade_by_model_name_only:
                position_adjustment = 0.1
            else:
                position_adjustment = 0.08

            va = 'bottom' if i % 2 == 0 else 'top'  # Vertical alignment
            ax.text((start + end) / 2, ax.get_ylim()[0] - ax.get_ylim()[1] * position_adjustment, model, ha="center", va=va, fontsize=10, color='black')

        # Add legend to plot explaining which hatch corresponds to which hardware setup
        handles = [mpl.patches.Patch(facecolor='white', edgecolor='black', hatch=hatch, label=hardware) for hardware, hatch in hardware_hatches.items()]
        legend_title = "Hardware"

        # Add legend to plot explaining which color corresponds to which promptset
        if promptset_colors:
            handles_promptset = [mpl.patches.Patch(color=color, label=promptset) for promptset, color in promptset_colors.items()]
            handles += handles_promptset
            legend_title += "/Promptset"

        # Add both handles to legend
        if showlegend:
            plt.legend(handles=handles, loc='upper left', title=legend_title, bbox_to_anchor=(1, 1))

        if shade_by_model_name_only:
            # Get the model sizes from the dataset names
            model_sizes = [name.split("_")[1] for name in dataset_names]

            # Set the xticklabels to only display the model sizes
            ax.set_xticklabels(model_sizes)
        else:
            # If shading is done by specific model AND size, then no xticks are needed.
            ax.set_xticklabels([])
            # Remove xticks (labels and tick marks)
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # Add labels and title
        ax.set_ylabel(column + " (kWh)")

        plt.tight_layout()

        plot_filename = f"boxplot_comparison_{column}"
        if filter_model_size:
            plot_filename += "_large_models_removed"

        plt.xticks(rotation=45, ha='right')
        plt.savefig(config.PLOTS_DIR_PATH / f"{plot_filename}.pdf", bbox_inches='tight')
        plt.show()

def analyze_per_promptset(datasets_dict):
    """Analyze the datasets per promptset.

    Find:
    - Mean energy consumption per prompt in each promptset
    - Mean response length in each promptset

    Prints out the results in this latex table format:
    Prompt dataset & Avg. energy cons. (kWh) & Avg. response token length \\

    """
    promptsets = set([dataset["promptset"] for dataset in datasets_dict.values()])

    for promptset in promptsets:
        datasets_in_promptset = [dataset for dataset in datasets_dict.values() if dataset["promptset"] == promptset]
        energy_consumptions = [dataset["dataset"].statistics.loc["energy_consumption_llm", "mean"] for dataset in datasets_in_promptset]
        response_lengths = [dataset["dataset"].statistics.loc["response_token_length", "mean"] for dataset in datasets_in_promptset]

        # Compute the averates across all datasets in the promptset
        avg_energy_consumption = np.mean(energy_consumptions)
        avg_response_length = np.mean(response_lengths)

        # Print the results
        print(f"& {promptset} & {avg_energy_consumption:.2e} & {avg_response_length:.2f} \\\\")

def analyze_per_model(datasets_dict):
    """Analyze the datasets per model (model name and size).

    Find:
    - Mean energy consumption per prompt in each model
    - Mean response length in each model

    Prints out the results in this latex table format:

    Model & Avg. energy cons. (kWh) & Avg. response token length \\

    """
    model_names = set([dataset["model_name_and_size"] for dataset in datasets_dict.values()])

    for model_name in model_names:
        datasets_in_model = [dataset for dataset in datasets_dict.values() if dataset["model_name_and_size"] == model_name]
        energy_consumptions = [dataset["dataset"].statistics.loc["energy_consumption_llm", "mean"] for dataset in datasets_in_model]
        response_lengths = [dataset["dataset"].statistics.loc["response_token_length", "mean"] for dataset in datasets_in_model]

        # Compute the averates across all datasets in the model
        avg_energy_consumption = np.mean(energy_consumptions)
        avg_response_length = np.mean(response_lengths)

        # Print the results
        print(f"& {model_name} & {avg_energy_consumption:.2e} & {avg_response_length:.2f} \\\\")

        
def generate_promptset_colors(datasets_dict):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    # Extract promptset from dataset names
    promptsets = set([dataset["promptset"] for dataset in datasets_dict.values()])
    # Sort promptsets to ensure consistent color assignment. Sort them reverse to get the same order as in the colors list
    promptsets = sorted(promptsets, reverse=True)
    # Make a dict matching promptset to color
    promptset_colors = {promptset: color for promptset, color in zip(promptsets, colors)}
    # Extract model names from dataset names

    return promptset_colors

def generate_hardware_hatches(datasets_dict, density=3):
    # Extract hardware from dataset names
    hardware_setups = set([dataset["hardware"] for dataset in datasets_dict.values()])

    # Make different hatch patters for different hardware setups, up to 10 different patterns
    hatches = ["|", "-", "x", "\\", "o", "O", ".", "/", "*", "+"]
    hatches = [hatch * density for hatch in hatches]
    # Match different hatches to hardware setups
    hardware_hatches = {hardware: hatch for hardware, hatch in zip(hardware_setups, hatches)}

    return hardware_hatches

def preprocess_datasets():

    datasets_path = config.DATA_DIR_PATH / "main_results"

    datasets = {}

    print("Preprocessing datasets...")
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

    # Extract model names from dataset names
    model_names = set([dataset["model_name"] for dataset in datasets.values()])
    # Extract hardware from dataset names
    hardware = set([dataset["hardware"] for dataset in datasets.values()])

    sorted_keys = sorted(
        datasets.keys(),
        key=lambda x: (x.split('_')[1], int(x.split('_')[2][:-1]))
    )
    datasets = {key: datasets[key] for key in sorted_keys}

    promptset_colors = generate_promptset_colors(datasets)
    hardware_hatches = generate_hardware_hatches(datasets)

    for dataset_name, dataset in datasets.items():
        dataset["color"] = promptset_colors[dataset["promptset"]]
        dataset["hatch"] = hardware_hatches[dataset["hardware"]]

    # Combine the dfs into one
    df = pd.concat([dataset["dataset"].df for dataset in datasets.values()])
    df.to_csv(config.COMPLETE_DATASET_PATH)

    print("Datasets preprocessed.")
    print("\n")

    return datasets, promptset_colors, hardware_hatches, df

def generate_statistics(datasets):

    # Iterate through the datasets and print number of samples in each, mean energy consumption, etc.
    print("Statistics for each dataset:")
    for dataset_name, dataset in datasets.items():
        # print(f"Dataset: {dataset_name}")
        dataset["dataset"].display_statistics()

    print("\n")

    print("Analysis per promptset:")
    analyze_per_promptset(datasets)
    print("\n")
    print("Analysis per model:")
    analyze_per_model(datasets)
    print("\n")

def generate_plots(datasets, promptset_colors, hardware_hatches):
    Dataset.boxplot_comparison(datasets, column="energy_consumption_llm", promptset_colors=promptset_colors, filter_model_size=True, showlegend=False, hardware_hatches=hardware_hatches)
    Dataset.boxplot_comparison(datasets, column="energy_per_token", promptset_colors=promptset_colors, filter_model_size=True, showlegend=True, hardware_hatches=hardware_hatches)

    models_to_include = [
            "alpaca_llama3_70b_server.csv",
            "alpaca_llama3_8b_laptop2.csv",
            "codefeedback_codellama_70b_workstation.csv"
    ]

    # Filter out models not in models_to_include
    datasets = {dataset_name: dataset for dataset_name, dataset in datasets.items() if dataset_name in models_to_include}

    Dataset.boxplot_comparison(datasets, column="energy_consumption_llm", promptset_colors=promptset_colors, filter_model_size=False, showlegend=False, hardware_hatches=hardware_hatches)
    Dataset.boxplot_comparison(datasets, column="energy_per_token", promptset_colors=promptset_colors, filter_model_size=False, showlegend=True, hardware_hatches=hardware_hatches)

def generate_subplots(datasets, promptset_colors, hardware_hatches, filter_model_size=False):
    Dataset.boxplot_comparison_subplots(datasets, column="energy_per_token",
                                        subplot_dimension="model_name_and_size",
                                        promptset_colors=promptset_colors,
                                        filter_model_size=filter_model_size)
    Dataset.boxplot_comparison_subplots(datasets, column="energy_consumption_llm",
                                        subplot_dimension="model_name_and_size",
                                        promptset_colors=promptset_colors,
                                        filter_model_size=filter_model_size)
    Dataset.boxplot_comparison_subplots(datasets, column="energy_per_token",
                                        subplot_dimension="hardware",
                                        promptset_colors=promptset_colors,
                                        filter_model_size=filter_model_size)
    Dataset.boxplot_comparison_subplots(datasets, column="energy_consumption_llm",
                                        subplot_dimension="hardware",
                                        promptset_colors=promptset_colors,
                                        filter_model_size=filter_model_size)

def make_forecasting_result_plot(filepath):

    df = pd.read_csv(filepath, index_col=0)
    # From column "dataset", remove the suffix ".csv"
    df['dataset'] = df['dataset'].apply(lambda x: x[:-4])

    # Sort the DataFrame by the R2 values
    df = df.sort_values(by='R2', ascending=False)

    print(df)

    # Make a new column that contains a string, which is the second word in the dataset column (each word is separated by an underscore)
    df['model'] = df['dataset'].apply(lambda x: x.split("_")[1])
    
    # Determine the clipping threshold for R2 values
    clip_value = -1.0  # This is an example; set the threshold based on your data analysis

    # Clip the R2 values at the threshold
    df['R2_clipped'] = df['R2'].apply(lambda x: max(x, clip_value))
    
    # Create a figure
    plt.figure(figsize=(7,3))

    # Create the bar plot with clipped R2 values. The bars are colored according to the model name
    bars = sns.barplot(x="R2_clipped", y="dataset", data=df, palette="viridis")
    # bars = sns.barplot(x="R2_clipped", y="dataset", data=df, palette="viridis", hue="model")


    # Highlight the clipped bars
    for bar, r2, r2_clipped in zip(bars.patches, df['R2'], df['R2_clipped']):
        if r2 < clip_value:
            bar.set_color((1,0,0,0.3))
            bar.set_edgecolor('black')
            # bar.set_hatch('//')
            plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     f'Clipped ({r2:.2f})', va='center', ha='left', color='black')

    # Loop through the bars and change color based on model
    for i, bar in enumerate(bars.patches):
        model = df['model'].iloc[i]
        color = sns.color_palette("viridis", len(df['model'].unique()))[list(df['model'].unique()).index(model)]
        bar.set_color(color)

    # Add legend for model colors
    handles = [mpl.patches.Patch(color=sns.color_palette("viridis", len(df['model'].unique()))[i], label=model) for i, model in enumerate(df['model'].unique())]
    plt.legend(handles=handles, title="LLM model type", loc='upper left')


    # Set axis labels and title
    plt.xlabel("R2 score")
    plt.ylabel("Energy consumption dataset")

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(str(config.PLOTS_DIR_PATH / filepath.stem) + ".pdf")
    plt.show()

def print_correlations_for_best_predictive_model(filepath):

    df = pd.read_csv(filepath, index_col=0)
    # From column "dataset", remove the suffix ".csv"
    df['dataset'] = df['dataset'].apply(lambda x: x[:-4])

    # Sort the DataFrame by the R2 values
    df = df.sort_values(by='R2', ascending=False)

    # Get the dataset with the highest R2 value
    best_dataset = df.iloc[0]['dataset']

    # Load the dataset
    dataset = Dataset(config.DATA_DIR_PATH / "main_results" / (best_dataset + ".csv"))

    # Perform correlation analysis
    Dataset.plot_single_correlations(dataset.df, num_corr=30)


def generate_expanded_statistics(datasets):

    # Iterate through the datasets and print number of samples in each, mean energy consumption, etc.
    print("Statistics for each dataset:")
    for dataset_name, dataset in datasets.items():
        print(f"Dataset: {dataset_name}")
        dataset["dataset"].display_expanded_statistics()
        print("\n")

if __name__ == '__main__':

    datasets, promptset_colors, hardware_hatches, df = preprocess_datasets()
    # generate_statistics(datasets)
    # generate_plots(datasets, promptset_colors, hardware_hatches)
    # generate_subplots(datasets, promptset_colors, hardware_hatches)
    # generate_subplots(datasets, promptset_colors, hardware_hatches, filter_model_size=True)
    # generate_expanded_statistics(datasets)
    # Dataset.plot_single_correlations(df, num_corr=30)
    make_forecasting_result_plot(config.DATA_DIR_PATH / "forecasting_results/forecasting_results_prompt.csv")
    # make_forecasting_result_plot(config.DATA_DIR_PATH / "forecasting_results/forecasting_results_response.csv")
    # print_correlations_for_best_predictive_model(config.DATA_DIR_PATH / "forecasting_results/forecasting_results_response.csv")
