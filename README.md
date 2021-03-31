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

Before attempting to install Kuiper, you must have the following:

* [Anaconda Package Manager](https://anaconda.org/)

Run the following commands to install kuiper and download the [datasets](https://www.dropbox.com/sh/lti7j1g4a1jgr4r/AAD802HuoxjZi8Xy7xXZbDs8a?dl=0).

```
git clone https://github.com/SymbioticLab/Kuiper
cd Kuiper
conda env create -f environment.yml # Install dependencies
conda activate kuiper
python setup.py install  # install kuiper
./data/download.sh -A    # download datasets 
```

# Run Experiments


# Validate Results

The output of the experiment will validate the following major claims in our paper:

#### 	**FL Training:**
* Kuiper outperforms existing random participant selection by 1.2×-14.1× in time-to-accuracy performance, while achieving 1.3%-9.8% better final model accuracy (§7.2.1) -> Table 1 and Figure 9.
* Kuiper achieves close-to-optimal model efficiency by adaptively striking the trade-off between statistical and system efficiency with different components (§7.2.2) -> Figure 11 and 12.
* Kuiper outperforms its counterpart over a wide range of parameters and different scales of experiments, while being robust to outliers (§7.2.3) -> Figure 13, 14.

#### 	**FL Testing:**
* Kuiper can serve developer testing criteria on data deviation while reducing costs by bounding the number of participants needed even without individual data characteristics(§7.3.1) —> Figure 16.
* With the individual information, Kuiper improves the testing duration by 4.7× w.r.t. Mixed Integer Linear Programming (MILP) solver, and is able to efficiently enforce developer preferences across millions of clients (§7.3.2) -> Figure 17.

## Training

Due to the great variety of training experiments, please follow the training  [README.md](https://github.com/SymbioticLab/Kuiper/blob/master/training/README.md) to initiate new training jobs and get performance results. As each experiment is really time-consuming, we strongly recommend the user to try (Yogi + ShuffleNet) setting on the OpenImage dataset, which is ***much faster*** than other datasets and strategies.

***Performance of model training (both accuracy and time-to-accuracy performance) often shows certain variations. We evaluate each setting over 5 runs and report the mean value in our paper.***

### Time to accuracy performance (Table 1 and Figure 9)

Please refer to ```training/evals/configs/DATA_NAME/conf.yml```. We spent > 3000 GPU hours to collection all results. 

### Performance breakdown (Figure 11 and Figure 12)

Please specify the following parameters in ```training/evals/configs/DATA_NAME/conf.yml``` to start the breakdown experiment:

    - Kuiper w/o Sys: - round_penalty: 0
    - Kuiper w/o Pacer: - pacer_step: 100000
    
### Sensitivity Analysis (Figure 13 and Figure 14)

Please specify different ```- round_penalty:``` (\alpha) and ```- total_worker: ``` (different number of participants K) in ```training/evals/configs/DATA_NAME/conf.yml```, and then submit jobs. 

## Testing

### Figure 16 - Preserving Data Representativeness 

```
cd testing
python plot_figure16.py     # few seconds
open figure16.pdf
```

This will produce plots close to Figure 16 (`figure/ref/figure16a.pdf` and `figure/ref/figure16b.pdf`) on page 12 of the paper. You might notice some variation compared to the original figure due to randomness of the experiments.

### Figure 17 - Enforcing Diverse Data Distribution 

Before running below script, you must install gurobi license by:

* Request an [academic license](https://www.gurobi.com/downloads/end-user-license-agreement-academic/) if possible. Otherwise, please contact us for a temporary license. 
* `grbgetkey [your license]` to install the license 
* TODO: maybe latex? sudo apt-get update and sudo apt-get install texlive-full

```
cd testing
python plot_figure17.py   # > 50 hours
# or python plot_figure17.py -k # ~ 1.5 hour
open figure17a.pdf figure17b.pdf
``` 

This will produce plots close to Figure 17 (`figure/ref/figure17a.pdf` and `figure/ref/figure17b.pdf`) on page 12 of the paper. You might notice some variation compared to the original figure as we removed a few long-running queries. 

Note: To save reviewers time, `python plot_figure17.py -k` will only run and plot the lines for Kuiper. We hope the runtime will convince you that MILP is extremely slow :).

# Repo Structure

```
Repo Root
|---- Training
    |---- evals     # Submit/terminate training jobs
        |---- configs   # Configuration examples
|---- Testing
|---- Data
       |---- download.sh   # Download all datasets     
|---- Kuiper          # Kuiper code base.
    
```

# Acknowledgements

Thanks to Qihua Zhou for his [Falcon repo](https://github.com/kimihe/Falcon).

# Contact
Fan Lai (fanlai@umich.edu) and Xiangfeng Zhu (xzhu0027@gmail.com)
