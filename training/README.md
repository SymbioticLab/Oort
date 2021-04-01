# Kuiper-Training

This repository contains scripts and instructions for reproducing the FL training experiments in our OSDI '21 paper.


Due to the great variety of training experiments, please follow the training  [README.md](https://github.com/SymbioticLab/Kuiper/blob/master/training/README.md) to initiate new training jobs and get performance results. As each experiment is really time-consuming, we strongly recommend the user to try (Yogi + ShuffleNet) setting on the OpenImage dataset, which is ***much faster*** than other datasets and strategies.

***Performance of model training (both accuracy and time-to-accuracy performance) often shows certain variations. We evaluate each setting over 5 runs and report the mean value in our paper.***

### Time to accuracy performance (Table 1 and Figure 9)

Please refer to ```training/evals/configs/DATA_NAME/conf.yml```. We spent > 3000 GPU hours to collection all results :). 

### Performance breakdown (Figure 11 and Figure 12)

Please specify the following parameters in ```training/evals/configs/DATA_NAME/conf.yml``` to start the breakdown experiment:  
Kuiper w/o Sys setting ```- round_penalty: 0```. Kuiper w/o Pacer setting: ```- pacer_step: 100000```

### Sensitivity Analysis (Figure 13, Figure 14 and Figure 15)

Please specify different ```- round_penalty:``` (\alpha for Figure 13) or ```- total_worker: ``` (different number of participants K for Figure 14) in ```training/evals/configs/DATA_NAME/conf.yml```, while keeping other configurations the same. 

Experiments of outliers are extremely slow as we need to get the final accuracy of the training, so we recommend the user to put this to the last. To run this, please first add ```- blacklist_rounds: 10``` to your configuration in order to enable the blacklist. Then specify different degrees of outliers ```- malicious_clients: 0.1``` (i.e., 10% clients are corrputed). 


# Preliminary

Our training evaluations rely on a distributed setting of ***multiple GPUs*** via the Parameter-Server (PS) architecture. 
In our paper, we used up to 68 GPUs to simulate the FL aggregation of 1300 participants in each round. 
Each training experiment is pretty time-consuming, as each GPU has to run multiple clients (1300/68) for each round. 

We outline some numbers on Tesla P100 GPU for each line in our plots when using 100 participants/round for reference (we also estimate some prices on [Google Cloud](https://cloud.google.com/products/calculator), but they are not accurate): 

| Setting      | Time to Target Accuracy  | Time to Converge |
| ----------- | ----------- | ----------- |
| Kuiper+YoGi      | 27  GPU hours (~$53)    |    58 GPU hours (~$111)   |
| YoGi             | 53  GPU hours (~$97)     |    121  GPU hours (~$230) |

Table 1: GPU hours on Openimage

***We recommend the developer to try OpenImage with YoGi setting, which is the most efficient setting.*** Instead, FedProx takes ~2X more GPU hours than YoGi, while the NLP task takes more than 4X GPU hours even with YoGi.  

***Note that the performance of model training (both accuracy and time-to-accuracy performance) often shows certain variations. We report the average results over 5 runs in our paper.***
We spent more than 6000 GPU hours on our evaluations.

# Getting Started 


## Setting GPU Cluster

***Please assure that these paths are consistent across all nodes so that the simulator can find the right path.***

- Make sure that the parameter-server node has access to other worker nodes via ```ssh```. 

- Follow [this](https://github.com/SymbioticLab/Kuiper/blob/master/README.md) to install all necessary libs, and download the training dataset.

Due to the high computation load on each GPU, we recommend the developer to make sure that each GPU is simulating no more than 20 clients. i.e., if the number of participants in each round is K, then we would better to use at least K/20 GPUs. 

## Setting Job Configuration

We provide an example of submitting a training job in ```/evals/manager.py```, whereby the user can submit her jobs on the PS node. 

- ```python manager.py submit conf.yml``` will submit a job with parameters specified in conf.yml on both the PS and worker nodes. 
We provide some example ```conf.yml``` in ```evals/configs``` for each dataset. 
They are close to the settings used in our evaluations. Comments in our example will help you quickly understand how to specify these parameters. 

- ```python manager.py stop job_name``` will terminate the running ```job_name``` (specified in yml) on the used nodes. 


***All logs will be dumped to ```log_path``` (specified in the configuration file) on each node. 
```training_perf``` locates at the PS node under this path, and the user can load it with ```pickle``` to check the time-to-accuracy performance.***
We also provide a script to plot this result ```python plot_openimage.py PATH_OF_training_perf_1 PATH_OF_training_perf_2```. This will save the performance in ```time_to_acc.pdf```.

This example is a temporary trial and we plan to optimize it soon. 
