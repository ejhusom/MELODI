# LLMEC – LLM Energy Consumption



## Getting started


1. Install Ollama.
2. Install Scaphandre.
3. Ensure that it is possible to read from the RAPL file (to measure power consumption) without root access ([CodeCarbon GitHub Issue #224](https://github.com/mlco2/codecarbon/issues/244)):

```
sudo chmod -R a+r /sys/class/powercap/intel-rapl
```
4. Run `run_experiment.py`.
