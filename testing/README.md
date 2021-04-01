# Kuiper-Testing

This folder contains scripts and instructions for reproducing the FL testing experiments in our OSDI '21 paper. 

NOTE: Before attempting to run testing script, you must download the datasets by running `./download.sh -A`.

### Figure 16 - Preserving Data Representativeness 

```
python plot_figure16.py     # few seconds
open figure16.pdf
```

This will produce plots close to Figure 16 (`Kuiper/figure/ref/figure16a.pdf` and `Kuiper/figure/ref/figure16b.pdf`) on page 12 of the paper. You might notice some variation compared to the original figure due to randomness of the experiments.

### Figure 17 - Enforcing Diverse Data Distribution 

Before running below script, you must install gurobi license by:

* Request an [academic license](https://www.gurobi.com/downloads/end-user-license-agreement-academic/) if possible. Otherwise, please contact us for a temporary license. 
* `grbgetkey [your license]` to install the license 
* [Optional] Install latex via: `sudo apt-get update and sudo apt-get install texlive-full`

```
cd testing
python plot_figure17.py   # > 50 hours
# or python plot_figure17.py -k # ~ 1.5 hour
open figure17a.pdf figure17b.pdf
``` 

This will produce plots close to Figure 17 (`Kuiper/figure/ref/figure17a.pdf` and `Kuiper/figure/ref/figure17b.pdf`) on page 12 of the paper. You might notice some variation compared to the original figure as we removed a few long-running queries. 

Note: To save reviewers time, `python plot_figure17.py -k` will only run and plot the lines for Kuiper. We hope the runtime will convince you that MILP is extremely slow :).