# FAITH Framework - Code Overview

This repository contains the source code for the **FAITH** framework. Below is an overview of its structure and instructions for running experiments.

## Repository Structure
FAITH/ # Main directory containing the implementation │── FAITH_ind/ # Code for inductive learning │── FAITH_trans/ # Code for transductive learning │── param/ # Stores most of the hyperparameters


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

(*Replace TBD with actual values.*)
