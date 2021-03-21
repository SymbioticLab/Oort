# Kuiper-Training

This repository contains scripts and instructions for reproducing the FL training experiments in our OSDI '21 paper.

# Preliminary

Our training evaluations rely on a distributed setting of multiple GPUs via the Parameter-Server (PS) architecture. In our paper, we used up to 68 GPUs to simulate the FL aggregation of 1300 participants in each round. Each training experiment is time-consuming (on  Tesla P100 GPU for each line in our plots when using 100 participants/round):

- OpenImage: ~ 140 GPU hours to reach target accuracy;

- Language Modeling: ~ 400 GPU hours to reach target accuracy;


# Getting Started 


## Setting GPU Cluster

- Make sure that the parameter-server node has access to other worker nodes via ```ssh```. 

- Download our repository on all nodes, and install all necessary libs.

- Download the training dataset.

Note that these paths should be consistent on all nodes.

## Setting Job Configuration

We provide an example of submitting a training job in ```\evals\submit.py```, whereby the user can submit her jobs on the PS node. User can specify different parameter settings in ```\evals\conf.yaml```, and then run ```python submit.py```. This will automatically create the worker processes on both the PS and worker nodes.

This example is a temporary trial and we plan to optimize it soon. 