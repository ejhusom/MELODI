#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare results from different monitoring tools.

Author:
    Erik Johannes Husom

Created:
    2024-10-04

"""
import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd

from EnergyDataset import EnergyDataset

def plot_results(df):
    # Boxplot comparing the total energy consumption measured by scaphandre, pyjoules and codecarbon for the various devices.
    fig, axs = plt.subplots(3, 1, figsize=(4, 7))

    df.boxplot(column=["energy_consumption_llm_cpu", "energy_consumption_llm_cpu_pyjoules", "energy_consumption_llm_cpu_codecarbon", "energy_consumption_llm_cpu_energymeter"], ax=axs[0])
    axs[0].set_title('CPU+RAM Energy Consumption')
    axs[0].set_ylabel('Energy (kWh)')
    axs[0].set_xticklabels(["MELODI", "PyJoules", "CodeCarbon", "EnergyMeter"])

    df.boxplot(column=["energy_consumption_llm_gpu", "energy_consumption_llm_gpu_pyjoules", "energy_consumption_llm_gpu_codecarbon", "energy_consumption_llm_gpu_energymeter"], ax=axs[1])
    axs[1].set_title('GPU Energy Consumption')
    axs[1].set_ylabel('Energy (kWh)')
    axs[1].set_xticklabels(["MELODI", "PyJoules", "CodeCarbon", "EnergyMeter"])

    df.boxplot(column=["energy_consumption_llm_total", "energy_consumption_llm_total_pyjoules", "energy_consumption_llm_total_codecarbon", "energy_consumption_llm_total_energymeter"], ax=axs[2])
    axs[2].set_title('Total Energy Consumption')
    axs[2].set_ylabel('Energy (kWh)')
    axs[2].set_xticklabels(["MELODI", "PyJoules", "CodeCarbon", "EnergyMeter"])

    plt.tight_layout()
    plt.savefig("energy_consumption_comparison.png")
    plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Compare energy monitoring tools.')
    parser.add_argument('dataset', type=str, help='Filepath to dataset')
    parser.add_argument('-m', '--model_name', type=str, help='Name of model', default=None)
    parser.add_argument('-s', '--model_size', type=int, help='Size of model', default=None)
    parser.add_argument('-p', '--promptset', type=str, help='Promptset', default=None)
    parser.add_argument('-hw', '--hardware', type=str, help='Hardware', default=None)

    args = parser.parse_args()

    # dataset = EnergyDataset(args.dataset, modelargs.model_name, args.model_size, args.promptset, args.hardware)
    dataset = EnergyDataset(args.dataset, 
                            model_name=args.model_name, 
                            model_size=args.model_size, 
                            promptset=args.promptset, 
                            hardware=args.hardware)
    print(dataset.metadata)



