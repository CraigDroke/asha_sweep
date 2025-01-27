# Hyperband-ASHA Sweep Script

## Overview
The Hyperband-ASHA Sweep Script is a Python script for hyperparameter optimization supporting GPU parallelism. It uses the Adaptive Successive Halving Algorithm (ASHA) combined with Hyperband to efficiently allocate resources for evaluating configurations.

---

## Table of Contents
1. [Explaination](#explaination)
2. [Installation](#installation)
3. [Key Features](#key-features)
4. [Usage](#usage)
5. [Class Details](#class-details)
6. [Class Methods Overview](#class-methods-overview)
7. [Non Class Methods Overview](#non-class-methods-overview)

---

## Explaination
Read the following 2 papers:
1. ASHA - https://arxiv.org/pdf/1810.05934
2. Hyperband - https://arxiv.org/abs/1603.06560

This script leverages the Hyperband model with ASHA as the scheduling algorithm to optimize hyperparameters efficiently. It combines the resource allocation strategy of Hyperband with the early stopping capability of ASHA to identify high-performing configurations while minimizing computational cost.

---

## Installation
To use this script:

1. Install the required dependencies:
    ```bash
    pip install requirements.txt
    ```
2. Ensure that pytorch is installed correctly with CUDA enabled and working

---

## Key Features

- **Hyperband-ASHA Algorithm**: Efficiently evaluates hyperparameter configurations.
- **GPU Management**: Utilizes multiple GPUs using subproceses and csv job regulation.
- **Custom Search Space**: Define hyperparameter search spaces in YAML format.
- **Dynamic Configuration Sampling**: Samples and validates unique configurations.
- **Bracket Initialization**: Dynamically calculates and initializes brackets for the optimization process.

---

## Usage

1. **Define Search Space**:
   Create a YAML file specifying the hyperparameter search space.

2. **Initialize the Script**:
   The following is an example of how to run the script with a custom
   ```python

   hyperband = HyperbandASHA(
       venv_path="/path/to/venv",
       evaluate_script="/path/to/evaluate.py",
       config_path="/path/to/config.yaml",
       save_path="/path/to/save/results",
       max_resource=81,
       reduction_factor=4,
       gpu_workers=[0, 1, 2, 3],
       num_runs_per_gpu=2, #NOTE: Watch out for high memory usage
       time_between_runs=10
   )
   ```

3. **Examine the Results**:
The following is the structure of save path
```python
    TODO
```

---

## Class Details

### `HyperbandASHA`

#### Parameters:
- **`venv_path`**: Path to the virtual environment used for evaluation.
- **`evaluate_script`**: Path to the evaluation script.
- **`config_path`**: Path to the YAML configuration file defining the search space.
- **`save_path`**: Directory for saving results.
- **`max_resource`**: Maximum resources allocated per configuration (default: `81`). This will affect the number of brackets and rungs.
- **`reduction_factor`**: Reduction factor (η) (default: `4`). This will affect the number of configs that get promoted and those that are removed
- **`gpu_workers`**: List of GPU indices available for execution (default: `[0, 1, 2, 3, 4, 5]`).
- **`num_runs_per_gpu`**: Maximum concurrent runs per GPU (default: `1`).
- **`time_between_runs`**: Delay between starting new runs in seconds (default: `10`).

---

## Class Methods Overview

### 1. **`reserve_gpus`**
Reserves GPUs by creating and holding placeholder tensors on the specified devices.

### 2. **`create_save_path`**
Creates the directory for saving results and initializes CSV files.

### 3. **`load_search_space`**
Loads the search space from a YAML configuration file.

### 4. **`sample_configuration`**
Randomly samples a configuration from the search space.

### 5. **`get_configurations`**
Generates unique configurations from the search space.

### 6. **`initialize_brackets`**
Initializes brackets for the ASHA algorithm by calculating configurations and resource allocation.

### 7. **`top_k_div_n_indices`**
Selects top configurations based on fitness scores.

### 8. **`get_job`**
Fetches the next configuration to execute from the current state of brackets.

### 9. **`get_available_gpus`**
Uses the nvidia-smi command to find a list of avaiable gpus (those with no processes running).

### 10. **`start_job`**
Checks to see if a trained model already exists and calls the evaulate function.

### 11. **`evaluate`**
Defines the command and sets up the subprocess (script is staged to run here).

### 12. **`run`**
Loops through all of rungs in each bracket to find the ideal configuration. This script calls almost all of the above functions and monitors the csv file to rapidly deploy new jobs as some complete.

### 13. **`get_best_configuration`**
Find the best configuration based on the fitness scores in the csv file.

---

## Non Class Methods Overview

### 1. **`get_fitness_score`**
Pull the fitness score from the supplied path and config name.

### 2. **`append_fitness_score`**
Add the supplied fitness score to the csv file in the proper row and column.

### 3. **`get_current_runs`**
Read in the supplied current_runs.csv file and retrieve a dictionary representation of the current runs active on each gpu.

### 4. **`modify_current_runs`**
Add, update, or remove a config from the current_runs.csv file.