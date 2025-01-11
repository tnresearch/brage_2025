# BRAGE benchmark
This is a dedicated framework aimed at testing ICL-capabilities in LLMs used to conduct the experiments in the NoDaLiDa submission: 

- *The BRAGE Benchmark: Evaluating Zero-shot Learning Capabilities of Large Language Models for Norwegian Customer Service Dialogues*

Please use the following citation:

``
bibtex citation goes here
``

## Benchmark environments and modes

- FP16 using **Transformers**:
    - **Chat**: Using the raw output from the model (with regex)
    - **Constrained**: Using outlines and FSM's to constrain the output to the target labels (classificati only)
- Quantization with **llama.cpp**:
    - **Chat**: Using the raw output from the model (with regex)

**Important note:** The capabilities of this source code goes beyond what has been done in the NoDaLiDa paper, please refer to the paper for the settings used.

# Docker
## Installing
- Clone this repo to a local folde
- Create new folder ``data/`` and ``models/``
- Add datasets (instructions to be added here ASAP) into folder: ``data/``
- Add GGUF files if any to ``models/``
- Update configuration files in the `configs/` directory:
    - `models.json` for any models added
    - `datasets.json` for any datasets added
    - `prompts.json` for any prompts added
- Build the docker image:
    - ``docker build -t brage .``

## Running a benchmark
- Navigate to cloned brage directory containing ``data/`` folder with tasks
- ``mkdir results``
- Run docker image:
    - **Transformers**:
      ``docker run -e HF_TOKEN="<your_token>" --gpus 1 --name brage --rm -v "$(pwd)":/workspace -it brage transformers python main.py --models google/gemma-2-2b-it --datasets Testdata --replications 1 --test_modes constrained --temperatures 0.0 --max_task_tokens 100 --refine False --seed True --suffix Experiment_name``
    - **Llama.cpp**:
      ``docker run -e HF_TOKEN="<your_token>" --gpus 1 --name brage --rm -v "$(pwd)":/workspace -it brage llama.cpp python main.py --models models/Meta-Llama-3-8B-Instruct.Q2_K.gguf --datasets Testdata --replications 1 --test_modes unconstrained --temperatures 0.0 --max_task_tokens 100 --refine False --seed True --suffix Experiment_name``


## Arguments:

- `--models`: List of model names to evaluate (must match `model_dest` in `models.json`)
- `--datasets`: List of dataset names to use (must match `dataset_name` in `datasets.json`)
- `--replications`: Number of replications (default: 10)
- `--test_modes`: List of test modes (e.g., "constrained", "unconstrained", default: ["unconstrained"])
- `--temperatures`: List of temperature values (default: [0.75])
- `--max_task_tokens`: List of maximum task token values (default: [2500])
- `--prompts`: List of prompts to use (default: [all prompts associated with the dataset])
- `--seed`: Whether to use deterministic sampling (True or False)
- `--suffix`: String to add to the foldername for the results (useful when running multiple tests in a row)




## Output

The framework will create a new directory in the `results/` folder for each run, containing:

- `design_table.csv`: The experimental design table
- `emissions.csv`: The emissions produced during the experiment
- `predictions.h5`: Raw prediction results on task(row)-level
- `agg_metrics.h5`: Aggregated performance metrics
- Copy of the configuration files
|
## Notes

- The capabilities of this source code goes beyond what has been done in the NoDaLiDa paper, please refer to the paper for the settings used.
- Incompatible model-prompt pairs and datasets will be automatically filtered out.
