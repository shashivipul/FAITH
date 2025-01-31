# FAITH Framework - Code Overview

This repository contains the source code for the **FAITH** framework. Below is an overview of its structure and instructions for running experiments.

## Repository Structure
FAITH/ # Main directory containing the implementation │── FAITH_ind/ # Code for inductive learning │── FAITH_trans/ # Code for transductive learning │── param/ # Stores most of the hyperparameters


## Running Experiments

### Transductive Setting
To run the FAITH framework in the transductive setting with a **SAGE** teacher on the **Cora** dataset, use the following command:

```bash
python3 train.py --teacher SAGE --exp_setting 0 --data_mode 0

