# LLM Energy Consumption Monitoring

Use local Large Language Models (LLMs) while monitoring energy usage. This project allows you to run prompts on LLMs and measure the energy consumption during the inference process.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Data](#data)
- [License](#license)
- [Contact Information](#contact-information)

## Installation

To install the required dependencies and set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ejhusom/llm-energy-consumption.git
   cd llm-energy-consumption
   ```

<!-- 2. **Create a virtual environment** (optional but recommended): -->
<!--    ```bash -->
<!--    python3 -m venv venv -->
<!--    source venv/bin/activate -->
<!--    ``` -->

<!-- 3. **Install the dependencies**: -->
<!--    ```bash -->
<!--    pip install -r requirements.txt -->
<!--    ``` -->

2. **Install [Ollama](https://ollama.com/)**.

3. **Install and configure `nvidia-smi` and `scaphandre` for monitoring** (if not already installed):
   - [nvidia-smi installation guide](https://developer.nvidia.com/nvidia-system-management-interface)
   - [scaphandre installation guide](https://github.com/hubblo-org/scaphandre)

4. **Ensure that it is possible to read from the RAPL file** (to measure power consumption) without root access ([CodeCarbon GitHub Issue #224](https://github.com/mlco2/codecarbon/issues/244)):
	```
	sudo chmod -R a+r /sys/class/powercap/intel-rapl
	```

5. **Ensure that no other processes than your LLM service are using the GPU**. If need be, move the display service to the integrated graphics:
	- `sudo nano /etc/X11/xorg.conf`
	- Paste:
	```bash
	Section "Device"
		Identifier "intelgpu0"
		Driver "intel"  # Use the Intel driver
	EndSection
	```
	- Restart display: `sudo systemctl restart display-manager`


## Usage

To run the script that prompts LLMs and monitors energy consumption, use the following command:

```
python3 LLMEC.py [PATH_TO_DATASET]
```

Or use the tool programmatically like this:

```python
from LLMEC import LLMEC

# Create an instance of LLMEC
llm_ec = LLMEC(config_path='path/to/config.ini')

# Run a prompt and monitor energy consumption
df = llm_ec.run_prompt_with_energy_monitoring(
    prompt="How can we use Artificial Intelligence for a better society?",
    save_power_data=True,
    plot_power_usage=True
)
```

## Configuration

The script uses a configuration file for various settings. The default configuration file path is specified in the `config` module. Below are some of the configurable options:

- `llm_service`: The LLM service to use (default: "ollama").
- `llm_api_url`: The API URL of the LLM service (default: "<http://localhost:11434/api/chat>").
- `model_name`: The model name for the request (default: "mistral").
- `verbosity`: Level of verbosity for logging (default: 0).

Example configuration (`config.ini`):

```ini
[General]
llm_service = ollama
llm_api_url = http://localhost:11434/api/chat
model_name = mistral
verbosity = 1
```

## Data

We have produced a dataset of energy consumption measurements for a diverse set of open-source LLMs.
This dataset is available at Hugging Face Datasets: [LLM Energy Consumption Dataset](https://huggingface.co/datasets/ejhusom/llm-inference-energy-consumption).

## Contributing

We welcome contributions! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Information

Maintained by Erik Johannes Husom. For any inquiries, please reach out via:

- Email: <erik.johannes.husom@sintef.no>
- GitHub: [ejhusom](https://github.com/ejhusom)
