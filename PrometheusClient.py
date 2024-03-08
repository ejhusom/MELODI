#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect metrics from Prometheus.

Author:
    Erik Johannes Husom

Created:
    2024-02-20

"""
from prometheus_api_client import PrometheusConnect
import datetime

class PrometheusClient:
    def __init__(self, url="http://localhost:9090"):
        self.url = url
        self.prom = PrometheusConnect(url=url, disable_ssl=True)

    def get_metric_range_data(self, metric_name, start_time, end_time):
        """Get measurements for a certain time range.

        Arguments:
            metric_name (str): A Prometheus query string for a given metric.
            start_time (datetime): Start of range.
            end_time (datetime): End of range.

        Returns:
            metric_data (list): A list, where each item of the list is a dict
                with metrics for a process that matched the query string given
                in 'metric_name'. An example of what each dict contains:
                >>> metric_data[0]['metric']
                {
                    '__name__': 'scaph_process_power_consumption_microwatts', 
                    'cmdline': 'ollamarunmistral', 
                    'exe': 'ollama', 
                    'instance': 'scaphandre:8080', 
                    'job': 'scaphandre', 
                    'pid': '3727'
                }
                >>> metric_data[0]['values']
                [[<TIMESTAMP>, <VALUE>], [<TIMESTAMP>, <VALUE>], ... ]

        """

        metric_data = self.prom.get_metric_range_data(
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time,
        )
        return metric_data

    def get_metric_aggregation(self, query, start_time, end_time, step='60s'):
        metric_data = self.prom.get_metric_aggregation(
            query=query,
            start_time=start_time,
            end_time=end_time,
        )
        return metric_aggregation

    def calculate_energy_consumption(self, metric_data):
        energy_consumption_joules = sum(
            [float(data['value'][1]) / 1e6 * 60 for metric in metric_data for data in metric['values']]
        )
        energy_consumption_wh = energy_consumption_joules / 3600
        return energy_consumption_joules, energy_consumption_wh

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    prom = PrometheusClient()
    query = 'scaph_process_power_consumption_microwatts{cmdline=~".*ollama.*"}'
    metric_name = 'scaph_process_power_consumption_microwatts{cmdline=~".*ollama.*"}'
    start_time = datetime.datetime.now() - datetime.timedelta(hours=1)  # 1 hour ago
    end_time = datetime.datetime.now()

    # metric_data = prom.get_metric_aggregation(
    #     query=query,
    #     start_time=start_time,
    #     end_time=end_time,
    # )

    # metric_data is a list
    metric_data = prom.get_metric_range_data(
        metric_name=metric_name,
        start_time=start_time,
        end_time=end_time,
    )

    timeseries = np.array(metric_data[0]["values"])
    timestamps = np.array([datetime.datetime.fromtimestamp(float(t)) for t in timeseries[:,0]])
    values = np.array(timeseries[:,1], dtype=np.int64)
    timestep = timestamps[1:] - timestamps[:-1]
    print(timestep)

    plt.plot(timestamps, values, ".-", label="values")
    # plt.plot(timestamps[1:], timestep, label="timesteps")
    plt.xlabel("time")
    plt.legend()
    plt.show()



    # energy_consumption_joules, energy_consumption_wh = prom.calculate_energy_consumption(metric_data)
    # print(f"Total energy consumption: {energy_consumption_joules:.2f} Joules ({energy_consumption_wh:.2f} Watt-hours)")
