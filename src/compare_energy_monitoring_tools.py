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
        "energy_consumption_llm_cpu": "energy_consumption_llm_cpu",
        "energy_consumption_llm_gpu": "energy_consumption_llm_gpu",
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
    df = merge_data_from_monitoring_tools()
    plot_results(df)
