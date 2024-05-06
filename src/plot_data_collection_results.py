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
from feature_extraction import extract_features


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
            # Convert the "created_at" column to datetime
            full_df['created_at'] = pd.to_datetime(full_df['created_at'])
            # Sort the DataFrame by the "created_at" column
            full_df = full_df.sort_values(by='created_at')
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

def plot_correlations(df, limit_columns=False):

    if limit_columns:
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
    else:
        corr_df = df

    corr = corr_df.corr(
        method="pearson",
        numeric_only=True,
    ).round(2)
    sns.heatmap(corr, annot=True, vmin=-1.0, vmax=1.0, cmap="RdBu")
    plt.tight_layout()
    plt.show()

def plot_single_correlations(df):
    # Calculate correlations with "energy_consumption_llm"
    correlations = df.corrwith(df["energy_consumption_llm"], numeric_only=True).sort_values(ascending=False)

    # Drop the "energy_consumption_llm" correlation with itself
    correlations.drop("energy_consumption_llm", inplace=True)

    # Create a bar plot for visualizing correlations
    plt.figure(figsize=(10, 8))
    sns.barplot(x=correlations.values, y=correlations.index, palette="viridis")
    plt.title("Feature Correlation with Energy Consumption of LLM")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Features")

    plt.show()

    #========================================================================================0

    import ast  # Import the Abstract Syntax Trees module to safely evaluate strings that contain Python literals

    def clean_category(value):
        try:
            # Attempt to evaluate the string as a Python literal (e.g., converting "['Remembering']" to ['Remembering'])
            evaluated_value = ast.literal_eval(value)
            if isinstance(evaluated_value, list):
                # If the evaluated value is a list, return its first element as the cleaned category
                return evaluated_value[0]
            else:
                # If it's not a list, return the value as it is
                return value
        except:
            # If there's any error in evaluating (meaning it's likely a normal string), just return the value
            return value

    # Apply the cleaning function to your 'TaskCategory' column
    df['type'] = df['type'].apply(clean_category)

    # Optionally, check for and convert any NaNs to a uniform representation (e.g., 'Unknown')
    df['type'] = df['type'].fillna('Unknown')

    import pandas as pd
    from scipy import stats
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # # Sample data: Replace this with your actual dataset
    # data = {
    #     'TaskCategory': ['Remembering', 'Understanding', 'Remembering', 'Applying', 'Understanding', 'Analyzing', 'Remembering'],
    #     'EnergyConsumption': [120.5, 110.2, 115.1, 130.2, 105.5, 125.3, 118.7]
    # }

    # df = pd.DataFrame(data)
    df['type'] = df['type'].astype(str)

    # It might also be a good idea to check for and drop any rows with missing values in these columns
    df.dropna(subset=['type', 'energy_consumption_llm'], inplace=True)


    # Perform ANOVA
    anova_results = ols('energy_consumption_llm ~ C(type)', data=df).fit()
    anova_table = sm.stats.anova_lm(anova_results, typ=2)
    print(anova_table)

    # If the p-value from ANOVA is significant, proceed with Tukey's HSD
    if anova_table['PR(>F)'][0] < 0.05:
        print("Significant differences found, proceeding with Tukey's HSD")
        tukey = pairwise_tukeyhsd(endog=df['energy_consumption_llm'], groups=df['type'], alpha=0.05)
        print(tukey)
    else:
        print("No significant differences found among categories.")


if __name__ == "__main__":
    print("Reading data...")
    df = read_data(force_concatenation=True)
    print(df.info())
    # # plot_data(df)
    # print("Extracting features...")
    # df_with_features = extract_features(df)
    # print("Plotting correlations...")
    # plot_single_correlations(df)
    # print("Saving data with extracted features...")
    # df_with_features.to_csv(config.MAIN_DATASET_WITH_FEATURES_PATH)
    # print("Done!")
