# FAITH Framework - Code Overview

This repository contains the source code for the **FAITH** framework. Below is an overview of its structure and instructions for running experiments.

## Repository Structure
FAITH/ # Main directory containing the implementation │── FAITH_ind/ # Code for inductive learning │── FAITH_trans/ # Code for transductive learning │── param/ # Stores most of the hyperparameters


## Requirements

Ensure you have the following dependencies installed before running the project:

```bash
pip install numpy==1.21.6 scipy==1.7.3 torch==1.6.0 dgl==0.6.1 scikit-learn==1.0.2

```


## Running Experiments

### Transductive Setting
To run the FAITH framework in the transductive (0)/ inductive(1) setting with a **SAGE** teacher on the **Cora** dataset, use the following command:

```bash
python3 train.py --teacher SAGE --exp_setting 0 --data_mode 0

python3 train.py --teacher SAGE --exp_setting 1 --data_mode 0

```





## Hyperparameters for Transductive Experiments

| Dataset      | Beta (β) | Gamma (γ) |
|-------------|---------|----------|
| **Cora**    | 2e-2     | 2e-5      |
| **Citeseer**| 1     | 1e-5      |
| **Pubmed**  | 2e-2     | 2e-6      |
| **A-Photo** | 2e-2     | 2e-7      |
| **CS**      | 2e-2     | 5e-7      |
| **Ogbn-arxiv** | 2e-2  | 1e-8      |

## Hyperparameters for Inductive Experiments

| Dataset      | Beta (β) | Gamma (γ) |
|-------------|---------|----------|
| **Cora**    | 2e-2     | 2e-4     |
| **Citeseer**| 1e-2     | 1e-4     |
| **Pubmed**  | 1e-1    | 1e-5     |
| **A-Photo** | 1e-1    | 6e-6   |
| **CS**      | 1e-1    | 1e-5    |


