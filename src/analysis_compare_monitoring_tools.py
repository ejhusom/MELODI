#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare results from different monitoring tools.

Author:
    Erik Johannes Husom

Created:
    2024-10-04

Usage:
    $ python analysis_compare_monitoring_tools.py dataset.csv --model_name "llama3.2" --model_size "1b" --promptset "alpaca" --hardware "laptop_lenovo"

"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import config
from EnergyDataset import EnergyDataset
import utils

# plt.style.use('seaborn-v0_8')
# plt.rcParams['boxplot.notch'] = True
# plt.rcParams['boxplot.showfliers'] = False

def compare_monitoring_tools_boxplots(dataset: EnergyDataset, per_token=False):
    """Compare energy consumption measured by different monitoring tools.

    Args:
        dataset (EnergyDataset): Dataset containing the data to compare.
        per_token (bool): If True, the energy consumption is divided by the
            number of tokens in the prompt.

    """

    # Get the data
    df = dataset.df

    if per_token:
        column_name = dataset.energy_per_token_column_name
    else:
        column_name = dataset.energy_consumption_llm_column_name

    # Boxplot comparing the total energy consumption measured by scaphandre,
    # pyjoules and codecarbon for the various devices.
    fig, axs = plt.subplots(3, 1, figsize=(4.8, 7))

    df.boxplot(column=[column_name + "_cpu",
                       column_name + "_cpu_pyjoules",
                       column_name + "_cpu_codecarbon",
                       column_name + "_cpu_energymeter"], ax=axs[0])

    axs[0].set_title('CPU Energy Consumption')
    axs[0].set_ylabel('Energy (kWh)')
    axs[0].set_xticklabels(["MELODI", "PyJoules", "CodeCarbon", "EnergyMeter"])

    df.boxplot(column=[column_name + "_gpu",
                       column_name + "_gpu_pyjoules",
                       column_name + "_gpu_codecarbon",
                       column_name + "_gpu_energymeter"], ax=axs[1])

    axs[1].set_title('GPU Energy Consumption')
    axs[1].set_ylabel('Energy (kWh)')
    axs[1].set_xticklabels(["MELODI", "PyJoules", "CodeCarbon", "EnergyMeter"])

    df.boxplot(column=[column_name + "_total",
                       column_name + "_total_pyjoules",
                       column_name + "_total_codecarbon",
                       column_name + "_total_energymeter"], ax=axs[2])

    axs[2].set_title('Total Energy Consumption')
    axs[2].set_ylabel('Energy (kWh)')
    axs[2].set_xticklabels(["MELODI", "PyJoules", "CodeCarbon", "EnergyMeter"])

    plt.tight_layout()
    plot_basename = f"tool_comparison_boxplot_{dataset.metadata['model_name']}_{dataset.metadata['model_size']}_{dataset.metadata['promptset']}_{dataset.metadata['hardware']}"
    if per_token:
        plot_basename += "_per_token"

    plot_basename = utils.clean_filename(plot_basename) 
    plt.savefig(config.PLOTS_DIR_PATH / f"{plot_basename}.pdf")
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def compare_monitoring_tools(dataset: EnergyDataset, per_token=False, plot_type='bar', log_scale=True):
    """Compare energy consumption measured by different monitoring tools.

    Args:
        dataset (EnergyDataset): Dataset containing the data to compare.
        per_token (bool): If True, the energy consumption is divided by the
            number of tokens in the prompt.
        plot_type (str): Type of plot to generate ('bar', 'violin', 'swarm', 'box').
        log_scale (bool): If True, the y-axis is logarithmic.

    """

    # Get the data
    df = dataset.df

    if per_token:
        column_name = dataset.energy_per_token_column_name
    else:
        column_name = dataset.energy_consumption_llm_column_name

    # Tools and energy consumption for CPU and GPU
    tools = ['MELODI', 'PyJoules', 'CodeCarbon', 'EnergyMeter']
    cpu_cols = [
        column_name + "_cpu",
        column_name + "_cpu_pyjoules",
        column_name + "_cpu_codecarbon",
        column_name + "_cpu_energymeter"
    ]
    gpu_cols = [
        column_name + "_gpu",
        column_name + "_gpu_pyjoules",
        column_name + "_gpu_codecarbon",
        column_name + "_gpu_energymeter"
    ]

    # Calculate means and standard deviations for CPU and GPU energy consumption
    cpu_means = df[cpu_cols].mean()
    gpu_means = df[gpu_cols].mean()
    cpu_std = df[cpu_cols].std()
    gpu_std = df[gpu_cols].std()

    # Organize data into a form that's easy to work with
    data = {
        'Tool': tools * 2,
        'Energy (kWh)': np.concatenate([cpu_means.values, gpu_means.values]),
        'Hardware': ['CPU'] * len(tools) + ['GPU'] * len(tools),
        'Std': np.concatenate([cpu_std.values, gpu_std.values])
    }
    plot_df = pd.DataFrame(data)

    # Plot based on the chosen type
    if plot_type == 'bar':
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a grouped bar chart (grouped by hardware, colors for tools)
        x = np.arange(2)  # Two groups: CPU and GPU
        width = 0.15  # Bar width

        for i, tool in enumerate(tools):
            ax.bar(x + i * width, plot_df[plot_df['Tool'] == tool]['Energy (kWh)'],
                   width, yerr=plot_df[plot_df['Tool'] == tool]['Std'], label=tool)

        if log_scale:
            ax.set_yscale('log')

        ax.set_xlabel('Hardware')
        ax.set_ylabel('Energy (kWh)')
        ax.set_xticks(x + width * (len(tools) - 1) / 2)
        ax.set_xticklabels(['CPU', 'GPU'])
        ax.legend(title='Tool')
        plt.tight_layout()

    elif plot_type == 'violin':
        sns.violinplot(x='Hardware', y='Energy (kWh)', hue='Tool', data=plot_df, split=True)

    elif plot_type == 'swarm':
        sns.swarmplot(x='Hardware', y='Energy (kWh)', hue='Tool', data=plot_df)

    elif plot_type == 'box':
        sns.boxplot(x='Hardware', y='Energy (kWh)', hue='Tool', data=plot_df)

    else:
        raise ValueError(f"Unknown plot_type '{plot_type}'. Use 'bar', 'violin', 'swarm', or 'box'.")

    # Set y-axis scale based on log_scale argument
    if log_scale and plot_type != 'bar':
        plt.yscale('log')

    # Save the plot
    plot_basename = f"tool_comparison_{plot_type}_graph_{dataset.metadata['model_name']}_{dataset.metadata['model_size']}_{dataset.metadata['promptset']}_{dataset.metadata['hardware']}"
    if per_token:
        plot_basename += "_per_token"

    plot_basename = utils.clean_filename(plot_basename)
    plt.savefig(config.PLOTS_DIR_PATH / f"{plot_basename}.pdf")
    plt.show()


def run_all_plot_options(dataset: EnergyDataset):
    """Run all combinations of plot_type and log_scale options.

    Args:
        dataset (EnergyDataset): Dataset containing the data to compare.

    """
    plot_types = ['bar', 'violin', 'swarm', 'box']
    log_scale_options = [True, False]

    for plot_type in plot_types:
        for log_scale in log_scale_options:
            print(f"Plotting {plot_type} with log_scale={log_scale}")
            compare_monitoring_tools(dataset, per_token=False, plot_type=f'{plot_type}', log_scale=log_scale)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Compare energy monitoring tools.')
    parser.add_argument('dataset', type=str, help='Filepath to dataset')
    parser.add_argument('-m', '--model_name', type=str, help='Name of model', default=None)
    parser.add_argument('-s', '--model_size', type=int, help='Size of model', default=None)
    parser.add_argument('-p', '--promptset', type=str, help='Promptset', default=None)
    parser.add_argument('-hw', '--hardware', type=str, help='Hardware', default=None)

    args = parser.parse_args()

    dataset = EnergyDataset(args.dataset, 
                            model_name=args.model_name, 
                            model_size=args.model_size, 
                            promptset=args.promptset, 
                            hardware=args.hardware)

    # compare_monitoring_tools_boxplots(dataset)
    # compare_monitoring_tools_boxplots(dataset, per_token=True)
    # compare_monitoring_tools_barchart(dataset)
    # compare_monitoring_tools_barchart(dataset, per_token=True)
    run_all_plot_options(dataset)
