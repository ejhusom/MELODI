# LLMEC – LLM Energy Consumption



## Getting started


1. Install Ollama.
2. Install Scaphandre.
3. Ensure that it is possible to read from the RAPL file (to measure power consumption) without root access ([CodeCarbon GitHub Issue #224](https://github.com/mlco2/codecarbon/issues/244)):

```
sudo chmod -R a+r /sys/class/powercap/intel-rapl
```
4. Ensure that no other processes than your LLM service are using the GPU.
5. Define path to dataset at the bottom of `LLMEC.py`.
6. Run `LLMEC.py`.
