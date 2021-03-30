# Kuiper-Training

This repository contains scripts and instructions for reproducing the FL training experiments in our OSDI '21 paper.

# Preliminary

Our training evaluations rely on a distributed setting of multiple GPUs via the Parameter-Server (PS) architecture. 
In our paper, we used up to 68 GPUs to simulate the FL aggregation of 1300 participants in each round. 
Each training experiment is pretty time-consuming, as each GPU has to run multiple clients (1300/68) for each round. 
We outline some numbers on Tesla P100 GPU for each line in our plots when using 100 participants/round for reference: 

- OpenImage: ~ 140 GPU hours to reach target accuracy;

- Language Modeling: ~ 400 GPU hours to reach target accuracy;


# Getting Started 


## Setting GPU Cluster

- Make sure that the parameter-server node has access to other worker nodes via ```ssh```. 

- Download our repository on all nodes, and install all necessary libs.

- Download the training dataset.

***Please assure that these paths are consistent across all nodes so that the simulator can find the right path.***

## Setting Job Configuration

We provide an example of submitting a training job in ```\evals\manager.py```, whereby the user can submit her jobs on the PS node. 

- ```python manager.py submit conf.yml``` will submit a job with parameters specified in conf.yml on both the PS and worker nodes. 
We provide some example ```conf.yml``` in ```evals\configs``` for each dataset. 
They are close to the settings used in our evalutions. Comments in our example will help you quickly understand how to specify these parameters. 
***All logs will be dumped to ```log_path``` specied in yml on each node. 
```training_perf``` locates at the PS node under this path, and the user can load it with ```pickle``` to check the time-to-accuracy performance。***

- ```python manager.py stop job_name``` will terminate the runnning ```job_name``` (specified in yml) on the used nodes. 

This example is a temporary trial and we plan to optimize it soon. 
