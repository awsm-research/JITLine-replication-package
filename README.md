
# JITLine-replication-package

  

This repository contains source code that we used to perform experiment in "JITLine: A Simpler, Better, Faster, Finer-grained Just-In-Time Defect Prediction" paper.


Please follow the steps below to reproduce the result

## Conda Environment Preparation

Run the following command in terminal (or command line) to prepare virtual environment

    git clone https://github.com/awsm-research/JITLine-replication-package.git
    cd ./JITLine-replication-package/JITLine/
    conda env create --file requirements.yml
    conda activate JITLine

## Experiment Result Replication Guide

### Additional dataset files

Download files from [Zenodo](https://zenodo.org/record/4596503#.ZFr1PexBxJE). (the dataset is stored in ./JITLine/data/). Then, put the dataset in ./JITLine/data/ of the cloned repository.

### Guideline to reproduce result of RQ1-RQ3

Open and run all cells in "JITLine_RQ1-RQ3.ipynb"  

### Guideline to reproduce result of RQ4

Note: To reproduce the result of this RQ, "JITLine_RQ1-RQ3.ipynb" must be run first

Open and run all cells in "JITLine_RQ4.ipynb" 
