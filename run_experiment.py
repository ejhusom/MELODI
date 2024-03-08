#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run experiments on LLMs and energy consumption.

Author:
    Erik Johannes Husom

Created:
    2024-02-15

"""
import datetime
import json
import subprocess
import sys

from LLMAPIClient import LLMAPIClient
from PrometheusClient import PrometheusClient

def run_prompt_with_energy_monitoring(
    model_name="mistral",
    prompt="How can we use Artificial Intelligence for a better society?",
    stream=False,
    save_data=False,
):
    """Demonstrates how to use the LLMAPIClient to send a request to the Ollama API.
    
    Args:
        model_name (str, optional): The model name for the request. Defaults to "mistral".
        prompt (str): The prompt to be sent.
        stream (bool, optional): Whether to stream the response. Defaults to False.
    """

    # LLM parameters
    llm_api_url = "http://localhost:11434/api/chat"
    ollama_client = LLMAPIClient(api_url=llm_api_url, model_name=model_name, role="user")

    # Prometheus parameters
    prom_client = PrometheusClient()
    metric_name = 'scaph_process_power_consumption_microwatts{cmdline=~".*ollama.*"}'

    # Prompt LLM
    data = ollama_client.call_api(prompt=prompt, stream=stream)
    end_time = datetime.datetime.now()
    if data:
        print(data)
    else:
        print("Failed to get a response.")
        sys.exit(1)

    metric_data = prom.get_metric_range_data(
        metric_name=metric_name,
        start_time=data["created_at"],
        end_time=end_time,
    )

    if save_data:
        timestamp_filename = data["created_at"].replace(':', '').replace('.', '')
        with open(f"llm_response_{timestamp_filename}.json", "w") as f:
            json.dump(data, f)

if __name__ == "__main__":

    run_prompt_with_energy_monitoring(prompt="What is the capital in France?", save_data=True)
