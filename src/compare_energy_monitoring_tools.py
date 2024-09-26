import matplotlib.pyplot as plt
import pandas as pd

# Read "data/dataset.csv" and "emissions.csv".
df = pd.read_csv("data/dataset.csv")
df_emissions = pd.read_csv("emissions.csv")

# From emissions, select only the columns "experiment_id", "cpu_energy", "gpu_energy", "ram_energy" and "energy_consumed".
df_emissions = df_emissions[["experiment_id", "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed"]]

# Merge codecarbon data into main dataframe, which now contains data from both scaphandre, pyjoules and codecarbon.
df = df.merge(df_emissions, left_on="prompt", right_on="experiment_id", how="left", suffixes=("", "_codecarbon"))

# Add suffix to indicate which measurements come from scaphandre and which come from codecarbon.
df = df.rename(columns={
    "energy_consumption_llm_cpu": "energy_consumption_llm_cpu_scaphandre",
    "energy_consumption_llm_gpu": "energy_consumption_llm_gpu_scaphandre",
    "energy_consumption_llm_total": "energy_consumption_llm_total_scaphandre",
    "cpu_energy": "cpu_energy_codecarbon",
    "gpu_energy": "gpu_energy_codecarbon",
    "ram_energy": "ram_energy_codecarbon",
    "energy_consumed": "energy_consumed_codecarbon"
})

# Save the resulting dataframe to "data/dataset_all_monitoring_tools.csv".
df.to_csv("data/dataset_all_monitoring_tools.csv", index=False)

# # Subplots, one for each energy source.
# fig, axs = plt.subplots(3, 1, figsize=(15, 15))
# df.plot(x="prompt", y=["energy_consumption_llm_cpu_scaphandre", "energy_consumption_llm_cpu_pyjoules", "cpu_energy_codecarbon"], kind="bar", ax=axs[0])
# df.plot(x="prompt", y=["energy_consumption_llm_gpu_scaphandre", "energy_consumption_llm_gpu_pyjoules", "gpu_energy_codecarbon"], kind="bar", ax=axs[1])
# df.plot(x="prompt", y=["energy_consumption_llm_total_scaphandre", "energy_consumption_llm_total_pyjoules", "energy_consumed_codecarbon"], kind="bar", ax=axs[2])
# plt.show()

# Boxplot comparing the total energy consumption measured by scaphandre, pyjoules and codecarbon for the various devices.
fig, axs = plt.subplots(3, 1, figsize=(7, 7))
df.boxplot(column=["energy_consumption_llm_cpu_scaphandre", "energy_consumption_llm_cpu_pyjoules", "cpu_energy_codecarbon"], ax=axs[0])
axs[0].set_title('CPU Energy Consumption')
axs[0].set_ylabel('Energy (kWh)')
axs[0].set_xticklabels(["Scaphandre", "PyJoules", "CodeCarbon"])
df.boxplot(column=["energy_consumption_llm_gpu_scaphandre", "energy_consumption_llm_gpu_pyjoules", "gpu_energy_codecarbon"], ax=axs[1])
axs[1].set_title('GPU Energy Consumption')
axs[1].set_ylabel('Energy (kWh)')
axs[1].set_xticklabels(["Scaphandre", "PyJoules", "CodeCarbon"])
df.boxplot(column=["energy_consumption_llm_total_scaphandre", "energy_consumption_llm_total_pyjoules", "energy_consumed_codecarbon"], ax=axs[2])
axs[2].set_title('Total Energy Consumption')
axs[2].set_ylabel('Energy (kWh)')
axs[2].set_xticklabels(["Scaphandre", "PyJoules", "CodeCarbon"])
plt.tight_layout()
plt.savefig("energy_consumption_comparison.png")
plt.show()
