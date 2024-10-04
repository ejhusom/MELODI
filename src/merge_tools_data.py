import matplotlib.pyplot as plt
import pandas as pd

def merge_data_from_monitoring_tools():
    # Read "data/dataset.csv" and "emissions.csv".
    df = pd.read_csv("data/dataset.csv")
    df_emissions = pd.read_csv("emissions.csv")

    # From emissions, select only the columns "experiment_id", "cpu_energy", "gpu_energy", "ram_energy" and "energy_consumed".
    df_emissions = df_emissions[["experiment_id", "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed"]]

    # Merge codecarbon data into main dataframe, which now contains data from both scaphandre, pyjoules and codecarbon.
    # df = df.merge(df_emissions, left_on="prompt", right_on="experiment_id", how="left", suffixes=("", "_codecarbon"))
    df = df.join(df_emissions)

    # Add suffix to indicate which measurements come from scaphandre and which come from codecarbon.
    df = df.rename(columns={
        "energy_consumption_llm_total": "energy_consumption_llm_total",
        "cpu_energy": "energy_consumption_llm_cpu_codecarbon",
        "gpu_energy": "energy_consumption_llm_gpu_codecarbon",
        "ram_energy": "energy_consumption_llm_ram_codecarbon",
        "energy_consumed": "energy_consumption_llm_total_codecarbon"
    })

    # Add ram_energy_codecarbon to the cpu_energy_codecarbon to get the total energy consumed by the system.
    df["energy_consumption_llm_cpu_codecarbon"] = df["energy_consumption_llm_cpu_codecarbon"] + df["energy_consumption_llm_ram_codecarbon"]

    # Save the resulting dataframe to "data/dataset_all_monitoring_tools.csv".
    df.to_csv("data/dataset_all_monitoring_tools.csv", index=False)

    return df

if __name__ == "__main__":
    df = merge_data_from_monitoring_tools()
