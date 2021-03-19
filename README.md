# Kuiper

This repository contains scripts and instructions for reproducing the experiments in our OSDI '21 paper "Efficient Federated Learning via Guided Participant Selection" (TODO: Provide a link to the final paper)

# Overview

* [Getting Started (x human-minutes + x compute-minutes)](#getting-started)
* [Run Experiments (x human-minutes + x compute-hour)](#run-experiments)
* [Validate Results (x human-minutes + x compute-minutes)](#validate-results)
* [Repo Structure](#repo-structure)
* [Contact](#contact)



# Getting Started 
Expected runtime: x human-minutes + x compute-minutes

Before attempting to install Kuiper, you must have the following installed:

* Python 3.7
* gurobipy 9.1.0
  * `conda install -c gurobi gurobi` ([alternative downloads](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-))
  * Request an [academic license](https://www.gurobi.com/downloads/end-user-license-agreement-academic/) if possible. Otherwise, please contact us for a temporary license. 
  * `grbgetkey [your license]` to install the license 
* PyTorch 1.1+ 
  * TODO: add instructions
* transformers xx
  * TODO: add instructions



download.sh (this will download and decompress ~76GB data to corresponding folders)

# Run Experiments

<!-- * Run `./exp.sh 6 30m 1` to run our tool on only *6 benchmarks* for *30 minutes each* with only *1 repetition*. 
  - This command takes only **3 hours** to run in total, and produces results that approximate the results shown in the paper.
  - Since there is only 1 repetition, there will be no error bars in the final plots.
  - Results will be saved in a directory called `results`.

* Run `./exp.sh 20 24h 10` to replicate the full experiments in the paper
  - This command takes **200 days** to run 10 reps of all 20 benchmarks for 24 hours each. 
  - Feel free to tweak the args to produce results with intermediate quality, depending on the time that you have.
  - Results will be saved in a directory called `results`. -->

# Validate Results

<!-- The output of the experiments will validate the following claims:
- Table 1: `results/tab1.csv` reproduces Table 1 on Page 5.
- Figure 2: `results/plot2.pdf` reproduces the plot in Figure 2 on Page 8.
- Page 7, para 3: "We outperform the baseline by 2x". See `results/comparison.csv`, where the second column (our performance) should have a value that is twice as much as the third column (baseline).

Our artifact does not validate the following claims:
- On Page 8, we say X, but this cannot be validated without access to specialized hardware/people, so we leave it out of scope of artifact evaluation. -->

# Repo Structure

```
Repo Root
|---- Training
|---- Testing
|---- Data
       |---- download.sh   # Download all datasets     
|---- Kuiper          # Kuiper code base.
    
```

# Acknowledgements

# Contact
Xiangfeng Zhu(xzhu0027@gmail.com) and Fan Lai(fanlai@umich.edu)