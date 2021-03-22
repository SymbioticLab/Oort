# Kuiper

This repository contains scripts and instructions for reproducing the experiments in our OSDI '21 paper "Efficient Federated Learning via Guided Participant Selection" (TODO: Provide a link to the final paper)

# Overview

* [Getting Started (30 human-minutes + 3 compute-hours)](#getting-started)
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

Run the following commands to install kuiper and download the [datasets](https://www.dropbox.com/sh/lti7j1g4a1jgr4r/AAD802HuoxjZi8Xy7xXZbDs8a?dl=0).

```
git clone https://github.com/SymbioticLab/Kuiper
cd Kuiper
python setup.py install  # install kuiper
./data/download.sh -A    # download all datasets (See ./download.sh -h on how to download a subset of datasets)
```

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

The output of the experiment will validate the following major claims in our paper:

#### 	**FL Training:**
* Kuiper outperforms existing random participant selection by 1.2×-14.1× in time-to-accuracy performance, while achieving 1.3%-9.8% better final model accuracy (§7.2.1).
* Kuiper achieves close-to-optimal model efficiency by adaptively striking the trade-off between statistical and system efficiency with different components (§7.2.2).
* Kuiper outperforms its counterpart over a wide range of parameters and different scales of experiments, while being robust to outliers (§7.2.3).
#### 	**FL Testing:**
* Kuiper can serve developer testing criteria on data deviation while reducing costs by bounding the number of participants needed even without individual data characteristics(§7.3.1).
* With the individual information, Kuiper improves the testing duration by 4.7× w.r.t. Mixed Integer Linear Programming (MILP) solver, and is able to efficiently enforce developer preferences across millions of clients (§7.3.2).

## Training

## Testing

### Figure 16 - Preserving Data Representativeness 

```
cd testing
python plot_figure16.py     # few seconds
```

This will produce plots(xxx) close to Figure 16 (`figure/ref/figure16a.png` and `figure/ref/figure16b.png`) on page 12 of the paper. You might notice some variation compared to the original figure due to random seeds.

### Figure 17 - Enforcing Diverse Data Distribution 

```
cd testing
python plot_figure17.py     # x seconds
```
This will produce plots(xxx) close to Figure 17 (`figure/ref/figure17a.png` and `figure/ref/figure17b.png`) on page 12 of the paper. 


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