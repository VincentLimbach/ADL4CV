# Project Overview

This repository contains the results of the joint research by Vincent Limbach and Leon Stadelmann conducted within the ADL4CV lecture. The project explores the creation of a merged neural field for object ensembles from the neural fields of individual objects. Our research defines two relevant generalization tasks and demonstrates that our approach successfully generalizes to unseen combinations and neural fields in 2D. For 3D, we show that while merging unseen combinations is feasible, handling unseen neural fields remains an open problem. This repository contains refactored code from the final stage of this project.

## Installation Instructions

To set up the project environment, follow these steps:

1. **Install Miniconda**:

   - Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html) suitable for your operating system.

2. **Create Python Environment**:

   - Open a terminal or command prompt.
   - Create a new conda environment with Python 3.11.5:
     ```sh
     conda create -n ADL4CV python=3.11.5
     ```
   - Activate the environment:
     ```sh
     conda activate ADL4CV
     ```

3. **Install Project Dependencies and Build**

First, navigate to your project directory. Then, follow these steps:

- **Install the `build` Package** (if not already installed):

  ```sh
  pip install build
  ```

- **Build the Project** using `python -m build`:

  ```sh
  python -m build
  ```

- **Add Repository to PATH**:
  ```sh
  export PYTHONPATH=.
  ```

## Usage Guide

The content of the project is distributed as follows:

- **/architectures** contains the Hypernetworks and NeuralFields files and classes
- **/data** contains the dataset classes. Furthermore, it and preprocessing scripts
- **/evaluation**: contains the scripts to create the figures
- **/models**: contains the pretrained models. We couldn't include the 3D due to size constraints
- **/utils**: contain files with reuable functionality

### Running the Scripts

You can execute these scripts from the command line without any arguments to use the default values as described in the report. The scripts will calculate and store the final validation prediction in the `ADL4CV/evaluation/latest_run` directory.

- To run the 2D Hypernetwork Trainer:

  ```sh
  python ADL4CV/HypernetworkTrainer2D.py
  ```

- To run the 3D Hypernetwork Trainer:
  ```sh
  python ADL4CV/HypernetworkTrainer3D.py
  ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
