#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run data collection for LLM energy usage.

Author:
    Erik Johannes Husom

Created:
    2024-06-08

"""
from LLMEC import LLMEC

class LLMECDataCollection():

    def __init__(self,
                 model_name="mistral",
         ):

        self.model_name = model_name

    def run_experiment(self):
        llm = LLMEC()

        dfs = []

        df = llm.run_prompt_with_energy_monitoring(
            model_name=self.model_name,
            prompt="What is the capital in France?", 
            save_data=True
        )

if __name__ == '__main__':

    dc = LLMECDataCollection()
    dc.run_experiment()
