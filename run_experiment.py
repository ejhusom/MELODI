#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run experiments on LLMs and energy consumption.

Author:
    Erik Johannes Husom

Created:
    2024-02-15

"""
import json

from LLMAPIClient import LLMAPIClient

def run_ollama_client(
    model_name="mistral",
    content="How can we use Artificial Intelligence for a better society?",
    stream=False,
    save_metadata=False,
):
    """Demonstrates how to use the LLMAPIClient to send a request to the Ollama API.
    
    Args:
        model_name (str, optional): The model name for the request. Defaults to "mistral".
        content (str): The content of the message to be sent.
        stream (bool, optional): Whether to stream the response. Defaults to False.
    """
    api_url = "http://localhost:11434/api/chat"
    ollama_client = LLMAPIClient(api_url=api_url, model_name=model_name, role="user")
    response, metadata = ollama_client.call_api(content=content, stream=stream)
    if response:
        print(response.text)
        print(metadata)
    else:
        print("Failed to get a response.")

    if save_metadata:
        timestamp_filename = metadata['created_at'].replace(':', '').replace('.', '')
        with open(f"llm_response_{timestamp_filename}.json", "w") as f:
            json.dump(metadata, f)

if __name__ == "__main__":

    run_ollama_client(content="What is the capital in France?", save_metadata=True)
    run_ollama_client(content="Write 10 paragraphs about a random topic.", save_metadata=True)
