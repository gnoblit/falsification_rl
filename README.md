# Falsification-Driven Exploration in Reinforcement Learning

This repository contains the implementation and evaluation of "Falsification-Driven Exploration," a novel reinforcement learning exploration strategy inspired by Karl Popper's philosophy of science.

The core idea is to reward an agent for designing and executing "experiments"—sequences of actions—that are explicitly intended to find flaws in (falsify) its own predictive model of the world.

This Falsification Agent is benchmarked against two standard baselines:
1.  **PPO (Proximal Policy Optimization)**: A strong policy-gradient algorithm with no intrinsic motivation.
2.  **Curiosity (ICM)**: A PPO agent augmented with an Intrinsic Curiosity Module, where the intrinsic reward is based on the prediction error of its forward dynamics model.

The primary testbed is the `MiniGrid-KeyCorridorS3R1-v0` environment, which features sparse rewards and requires hierarchical reasoning, making it an ideal challenge for advanced exploration strategies.

## Project Structure

```
├── README.md           # You are here!
├── ideas.md            # The detailed research proposal and technical breakdown
├── pyproject.toml      # Project metadata and dependencies for uv
├── main.py             # Main entry point for running experiments (using Hydra)
├── configs/            # Hydra configuration files for experiments
│   ├── agent/          # Agent-specific configs (ppo, curiosity, falsification)
│   ├── env/            # Environment configs
│   └── ...
└── falsify/            # Main source code package
    ├── agents/         # RL agent implementations
    ├── components/     # Reusable modules (Theory, Falsifier, etc.)
    ├── models.py       # Neural network architecture definitions
    ├── storage.py      # RolloutStorage for collecting experience
    └── training/
        └── trainer.py  # The main training and evaluation loop
```

## Setup

This project uses `uv` for fast and efficient Python environment and package management.

1.  **Install uv:**
    ```bash
    pip install uv
    ```

2.  **Create and activate the virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    # On Windows use: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The `-e` flag installs the project in "editable" mode, which is useful for development.
    ```bash
    uv pip install -e .
    ```

## Running Experiments

Experiments are managed with `hydra` and can be launched using the `run-experiment` script defined in `pyproject.toml`. You can override any configuration parameter from the command line.

**1. Run the Falsification Agent (Proposed Method):**
```bash
run-experiment agent=falsification exp_name="Falsify_KeyCorridor_Test"
```

**2. Run the Curiosity Agent (Baseline 1):**
```bash   
run-experiment agent=curiosity exp_name="Curiosity_KeyCorridor_Test"
```

**3. Run the Standard PPO Agent (Baseline 2):**
```bash      
run-experiment agent=ppo exp_name="PPO_KeyCorridor_Test"
```

### Configuration
You can modify the YAML files in the `configs/` directory to change hyperparameters, such as learning rates, environment settings, or agent-specific parameters, without altering the code.

## Analyzing Results

Training progress, performance metrics, and losses are logged to TensorBoard.

1.  **Launch TensorBoard:**
    ```bash
    tensorboard --logdir runs
    ```

2.  **View the Dashboard:**
    Navigate to `http://localhost:6006/` in your browser. You can directly compare the sample efficiency (`charts/episodic_return`), learning stability (`losses/*`), and intrinsic reward signals across the different experimental runs.